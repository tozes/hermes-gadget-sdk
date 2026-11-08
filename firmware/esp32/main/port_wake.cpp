// The wake word detector (CONFIG_HG_WAKE_WORD). While the core arms it (hg::WakeWord), the
// codec microphone's mic task hands it every 24 kHz chunk through a ring of whole chunks
// (hg::WakeFeed) instead of posting them to the app (CodecMic::set_listening), each tagged with
// the armed session it was heard in. Its own task resamples them to 16 kHz
// (hg::Resample3to2), runs TensorFlow Lite Micro's audio frontend (40 features per 30 ms
// window, 10 ms step) and the streaming microWakeWord model (one inference per 3 steps), and
// a mean of the last N probabilities above the model's cutoff (hg::WakeWindow) posts one
// EventType::Wake carrying that session, while it is still armed. The model, its phrase,
// cutoff, window and arena come from the sdkconfig (CONFIG_HG_WAKE_WORD_MODEL and its values,
// taken from the model's manifest); the arena gets CONFIG_HG_WAKE_WORD_ARENA_EXTRA on top for
// the scratch buffers esp-nn's kernels plan, which the manifest's size leaves out.
#include "port.hpp"  // first: pulls in FreeRTOS.h ahead of task.h/queue.h

#include "sdkconfig.h"

#if CONFIG_HG_WAKE_WORD

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>

#include "es7210_gain.hpp"
#include "esp_console.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "frontend.h"
#include "frontend_util.h"
#include "resample.hpp"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "wake_feed.hpp"
#include "wake_window.hpp"

namespace hgp::wake {
namespace {

const char* TAG = "hg.wake";

constexpr int kSampleRate = 16000;
constexpr int kFeatures = 40;
constexpr int kStepMs = 10;  // one feature row
// The frontend's uint16 outputs are scaled by this before quantising: microWakeWord's
// training features are the same outputs as floats (pymicro-features does the same).
constexpr float kFeatureScale = 0.0390625f;
constexpr int kRefractoryMs = 2000;  // after a detection, before the next can count
// Audio from the mic task, 24 kHz, in whole chunks (hg::WakeFeed): 37 of them, 0.49 s, in PSRAM.
using Feed = hg::WakeFeed<CodecMic::kChunk>;
constexpr size_t kFeedSlots = 38;  // one stays empty
// Inference runs on this stack, in internal RAM (TFLM's kernels keep their scratch there).
// ponytail: not measured on the board; `diag` `stacks` shows hg-wake's free bytes to trim it.
constexpr uint32_t kTaskStack = 8192;

// Puts wake/<file> in flash as <sym>_start .. <sym>_end, 16-byte aligned as TFLM
// wants its model. HG_WAKE_DIR comes from CMakeLists.txt.
#define HG_INCBIN(sym, file)                                                              \
  asm(".section .rodata." #sym ",\"a\"\n.balign 16\n.global " #sym "_start\n" #sym       \
      "_start:\n.incbin \"" HG_WAKE_DIR "/" file "\"\n.global " #sym "_end\n" #sym "_end:\n" \
      ".previous\n");                                                                     \
  extern const uint8_t sym##_start[] asm(#sym "_start");                                  \
  extern const uint8_t sym##_end[] asm(#sym "_end");

HG_INCBIN(hg_wake_model, CONFIG_HG_WAKE_WORD_MODEL ".tflite")
#if CONFIG_HG_WAKE_WORD_TEST_CLIP
HG_INCBIN(hg_wake_clip, CONFIG_HG_WAKE_WORD_MODEL "_test.pcm")
#endif

// The operators the built-in models use (checked with the LiteRT interpreter on the host;
// SPLIT_V is Hey Hermes only).
using Resolver = tflite::MicroMutableOpResolver<13>;
// Resource variable slots (VAR_HANDLE); both built-in models use 6.
constexpr int kVariables = 20;

// TFLM's frontend and the streaming model, from 16 kHz samples to probabilities.
class Model {
 public:
  // Whatever begin() got, including after it failed part way. Free(nullptr) is a no-op
  // throughout, and the frontend starts zeroed.
  ~Model() {
    delete interp_;  // before the arena it lives in
    heap_caps_free(arena_);
    FrontendFreeStateContents(&frontend_);
  }

  // Allocates everything once: the arena in PSRAM, the interpreter and the frontend.
  // On false the caller deletes the model; reset() and feed() are for a model that began.
  bool begin() {
    FrontendConfig fc;
    FrontendFillConfigWithDefaults(&fc);
    fc.window.size_ms = 30;
    fc.window.step_size_ms = kStepMs;
    fc.filterbank.num_channels = kFeatures;
    fc.filterbank.lower_band_limit = 125.0f;
    fc.filterbank.upper_band_limit = 7500.0f;
    fc.noise_reduction.smoothing_bits = 10;
    fc.noise_reduction.even_smoothing = 0.025f;
    fc.noise_reduction.odd_smoothing = 0.06f;
    fc.noise_reduction.min_signal_remaining = 0.05f;
    fc.pcan_gain_control.enable_pcan = 1;
    fc.pcan_gain_control.strength = 0.95f;
    fc.pcan_gain_control.offset = 80.0f;
    fc.pcan_gain_control.gain_bits = 21;
    fc.log_scale.enable_log = 1;
    fc.log_scale.scale_shift = 6;
    if (!FrontendPopulateState(&fc, &frontend_, kSampleRate)) {
      ESP_LOGE(TAG, "frontend setup failed");
      return false;
    }
    resolver_.AddVarHandle();
    resolver_.AddReadVariable();
    resolver_.AddAssignVariable();
    resolver_.AddCallOnce();
    resolver_.AddConcatenation();
    resolver_.AddConv2D();
    resolver_.AddDepthwiseConv2D();
    resolver_.AddFullyConnected();
    resolver_.AddLogistic();
    resolver_.AddQuantize();
    resolver_.AddReshape();
    resolver_.AddSplitV();
    resolver_.AddStridedSlice();

    arena_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, kArena, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena_) {
      ESP_LOGE(TAG, "no PSRAM for a %u B arena", unsigned(kArena));
      return false;
    }
    // The resource variables' buffers (the model's streaming state) come from the arena
    // whichever allocator holds their table (MicroResourceVariables::Allocate), so one arena.
    tflite::MicroAllocator* alloc = tflite::MicroAllocator::Create(arena_, kArena);
    tflite::MicroResourceVariables* vars = tflite::MicroResourceVariables::Create(alloc, kVariables);
    interp_ = new (std::nothrow) tflite::MicroInterpreter(tflite::GetModel(hg_wake_model_start), resolver_, alloc, vars);
    if (!interp_) {
      ESP_LOGE(TAG, "no memory for the interpreter");
      return false;
    }
    if (interp_->AllocateTensors() != kTfLiteOk) {
      ESP_LOGE(TAG, "AllocateTensors failed: arena %u B (manifest %u + extra %u), %u B taken when it stopped",
               unsigned(kArena), unsigned(CONFIG_HG_WAKE_WORD_ARENA), unsigned(CONFIG_HG_WAKE_WORD_ARENA_EXTRA),
               unsigned(interp_->arena_used_bytes()));
      return false;
    }
    arena_used_ = interp_->arena_used_bytes();
    TfLiteTensor* in = interp_->input(0);
    TfLiteTensor* out = interp_->output(0);
    if (in->type != kTfLiteInt8 || in->dims->size != 3 || in->dims->data[2] != kFeatures ||
        out->type != kTfLiteUInt8) {
      ESP_LOGE(TAG, "unexpected model tensors");
      return false;
    }
    stride_ = in->dims->data[1];
    in_scale_ = in->params.scale;
    in_zero_ = in->params.zero_point;
    out_scale_ = out->params.scale;
    out_zero_ = out->params.zero_point;
    ESP_LOGI(TAG, "%s: stride %d, arena %u of %u B (PSRAM, manifest %u + extra %u), %d resource variable slots in it",
             CONFIG_HG_WAKE_WORD_MODEL, stride_, unsigned(arena_used_), unsigned(kArena),
             unsigned(CONFIG_HG_WAKE_WORD_ARENA), unsigned(CONFIG_HG_WAKE_WORD_ARENA_EXTRA), kVariables);
    return true;
  }

  // The tensor arena, for `diag`: its size and what AllocateTensors took of it.
  static constexpr size_t kArena = CONFIG_HG_WAKE_WORD_ARENA + CONFIG_HG_WAKE_WORD_ARENA_EXTRA;
  size_t arena_used() const { return arena_used_; }

  // Milliseconds of audio per probability.
  int ms_per_inference() const { return stride_ * kStepMs; }

  // Clears the audio and model streaming state.
  void reset() {
    FrontendReset(&frontend_);
    interp_->Reset();
    frames_ = 0;
  }

  // Feeds 16 kHz samples and calls on_probability(p) after each inference. Allocation-free.
  // False if an inference failed.
  template <class F>
  bool feed(const int16_t* pcm, size_t n, F&& on_probability) {
    while (n) {
      size_t read = 0;
      FrontendOutput fo = FrontendProcessSamples(&frontend_, pcm, n, &read);
      pcm += read;
      n -= read;
      if (!fo.size) continue;
      int8_t* row = interp_->input(0)->data.int8 + frames_ * kFeatures;
      for (int i = 0; i < kFeatures; ++i) {
        long q = std::lround(fo.values[i] * kFeatureScale / in_scale_) + in_zero_;
        row[i] = int8_t(q < -128 ? -128 : q > 127 ? 127 : q);
      }
      if (++frames_ < stride_) continue;
      frames_ = 0;
      const int64_t t0 = esp_timer_get_time();
      if (interp_->Invoke() != kTfLiteOk) return false;
      const uint32_t us = uint32_t(esp_timer_get_time() - t0);
      ++invokes_;
      invoke_us_total_ += us;
      if (us > invoke_us_max_.load(std::memory_order_relaxed)) invoke_us_max_.store(us, std::memory_order_relaxed);
      invoke_us_mean_.store(uint32_t(invoke_us_total_ / invokes_), std::memory_order_relaxed);
      on_probability((float(interp_->output(0)->data.uint8[0]) - out_zero_) * out_scale_);
    }
    return true;
  }

  // Inference time since boot, for `diag` (written here, read by the app task).
  std::atomic<uint32_t> invoke_us_mean_{0}, invoke_us_max_{0};

 private:
  FrontendState frontend_{};
  Resolver resolver_;
  uint8_t* arena_ = nullptr;  // holds the allocator and resource variables too
  size_t arena_used_ = 0;
  tflite::MicroInterpreter* interp_ = nullptr;
  float in_scale_ = 1, out_scale_ = 1;
  int in_zero_ = 0, out_zero_ = 0;
  int stride_ = 1, frames_ = 0;
  uint32_t invokes_ = 0;
  uint64_t invoke_us_total_ = 0;
};

// The port's hg::WakeWord: arming switches the mic task's capture and restarts the detector.
class Detector final : public hg::WakeWord {
 public:
  // Only for a detector whose begin() failed: one that began has a task using it for good.
  ~Detector() override { heap_caps_free(ring_); }

  // Everything that can fail comes before the task starts; after that the detector is
  // published (instance, the mic's sink) and never freed. On false the caller deletes it.
  bool begin(CodecMic& mic) {
    if (!model_.begin()) return false;
    window_ = hg::WakeWindow(CONFIG_HG_WAKE_WORD_CUTOFF_PCT / 100.0f, CONFIG_HG_WAKE_WORD_WINDOW,
                             kRefractoryMs / model_.ms_per_inference());
    if (!ring_ || xTaskCreate(&Detector::task, "hg-wake", kTaskStack, this, 3, &task_) != pdPASS) {
      ESP_LOGE(TAG, "no memory for the detector's audio buffer or task");
      return false;
    }
    instance = this;  // before the sink, which reaches it from here
    mic_ = &mic;
    mic.set_sink(&Detector::sink);
    ESP_LOGI(TAG, "ready: \"%s\", cutoff %d%%, window %d", CONFIG_HG_WAKE_WORD_PHRASE, CONFIG_HG_WAKE_WORD_CUTOFF_PCT,
             window_.window());
    return true;
  }

  // The detector never reads the ES7210 itself: the mic task is its only reader and hands
  // the detector chunks only while listening and not capturing (hg::MicPower::turn). So
  // disarming just clears listening, and the AudioIn::start() the core calls next takes
  // every chunk from then on even if the mic task hasn't seen this yet. The session goes to
  // the feed before listening changes, so chunks the mic task hands over from then on carry
  // it: audio still queued from an earlier session is skipped, a detection from it posts
  // nothing (hg::WakeFeed::current), and the core ignores one posted before (App::on_wake).
  void set_armed(bool armed) override {
    ++session_;
    feed_.set_session(armed ? session_ : 0);
    mic_->set_listening(armed);
    ESP_LOGI(TAG, "%s (session %u)", armed ? "armed" : "disarmed", unsigned(session_));
  }
  uint32_t session() const override { return session_; }

  void report(hg::json::Value& wake) const {
    wake.set("last_peak", static_cast<double>(last_peak_.load()))
        .set("dropped_chunks", static_cast<unsigned>(feed_.dropped()))
        .set("invoke_us_mean", static_cast<unsigned>(model_.invoke_us_mean_.load()))
        .set("invoke_us_max", static_cast<unsigned>(model_.invoke_us_max_.load()))
        .set("arena_bytes", static_cast<unsigned>(Model::kArena))
        .set("arena_used_bytes", static_cast<unsigned>(model_.arena_used()))
        .set("model", CONFIG_HG_WAKE_WORD_MODEL);
  }

  // `wake-test`: runs on the detector's task, between chunks, so never alongside live
  // detection on the same model. False while one is queued or running (until it has printed).
  bool request_test() {
    bool idle = false;
    return test_.compare_exchange_strong(idle, true);
  }

  static Detector* instance;

 private:
  // On the mic task: a chunk that doesn't fit is dropped whole (the detector fell 0.49 s
  // behind), and the detector starts its audio over at the next one.
  static void sink(const int16_t* pcm) {
    if (instance->feed_.push(pcm)) xTaskNotifyGive(instance->task_);
  }

  static void task(void* arg) {
    auto* self = static_cast<Detector*>(arg);
    for (;;) {
      // A second without audio (disarmed) still comes round for `wake-test`.
      if (!self->feed_.take([self](const int16_t* pcm, size_t n, uint32_t session, bool restart) {
            self->hear(pcm, n, session, restart);
          }))
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
      if (self->test_.load()) {
        self->run_test();
        self->feed_.restart();  // live detection starts over from the next chunk
        self->test_ = false;    // printed: the next wake-test may queue
      }
    }
  }

  // One chunk heard in armed session `session`; `restart`: it doesn't follow the last one.
  void hear(const int16_t* in, size_t samples, uint32_t session, bool restart) {
    if (restart) {
      resampler_.reset();
      model_.reset();
      window_.reset();
    }
    int16_t pcm[hg::Resample3to2::max_out(CodecMic::kChunk)];
    const size_t n = resampler_.process(in, samples, pcm);
    bool detected = false;
    const bool ok = model_.feed(pcm, n, [this, &detected](float p) {
      if (window_.push(p)) detected = true;
    });
    if (!ok && !failed_) ESP_LOGE(TAG, "inference failed");
    failed_ = !ok;
    if (!detected) return;
    // The window has started over; the model starts from silence too.
    model_.reset();
    const float mean = window_.mean();
    last_peak_ = mean;
    if (!feed_.current(session)) return;  // disarmed or re-armed while this audio waited
    ESP_LOGI(TAG, "detected p=%.2f", static_cast<double>(mean));
    events::post(EventType::Wake, nullptr, 0, session);
  }

  // Feeds the built-in 16 kHz clip (or a second of silence) through the model and the
  // detection rule, and prints what it saw. Live detection restarts afterwards.
  void run_test() {
#if CONFIG_HG_WAKE_WORD_TEST_CLIP
    const int16_t* clip = reinterpret_cast<const int16_t*>(hg_wake_clip_start);
    const size_t samples = size_t(hg_wake_clip_end - hg_wake_clip_start) / sizeof(int16_t);
#else
    const int16_t* clip = nullptr;
    const size_t samples = kSampleRate;
#endif
    static const int16_t kSilence[kSampleRate / 100] = {};
    model_.reset();
    window_.reset();
    float max_p = 0, max_mean = 0;
    uint32_t inferences = 0, detected_at = 0;
    bool ok = true;
    for (size_t off = 0; off < samples && ok; off += kSampleRate / 100) {
      const size_t n = samples - off < kSampleRate / 100 ? samples - off : kSampleRate / 100;
      ok = model_.feed(clip ? clip + off : kSilence, n, [&](float p) {
        ++inferences;
        if (p > max_p) max_p = p;
        const bool hit = window_.push(p);
        if (window_.mean() > max_mean) max_mean = window_.mean();
        if (hit && !detected_at) detected_at = inferences;
      });
    }
    std::printf("wake-test %s on %s (%u ms): max p %.3f, max mean %.3f, cutoff %.2f, %s at %u ms; "
                "%u inferences; since boot %u us mean, %u us max%s\n",
                CONFIG_HG_WAKE_WORD_MODEL, clip ? "clip" : "silence", unsigned(samples * 1000 / kSampleRate),
                static_cast<double>(max_p), static_cast<double>(max_mean), CONFIG_HG_WAKE_WORD_CUTOFF_PCT / 100.0,
                detected_at ? "DETECTED" : "not detected", unsigned(detected_at * model_.ms_per_inference()),
                unsigned(inferences), unsigned(model_.invoke_us_mean_.load()), unsigned(model_.invoke_us_max_.load()),
                ok ? "" : "; inference FAILED");
  }

  Model model_;
  hg::Resample3to2 resampler_;
  hg::WakeWindow window_{1, 1, 0};
  // feed_'s chunks, in PSRAM (null if that failed; begin() checks).
  Feed::Slot* ring_ = static_cast<Feed::Slot*>(
      heap_caps_malloc(sizeof(Feed::Slot) * kFeedSlots, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  Feed feed_{ring_, kFeedSlots};
  TaskHandle_t task_ = nullptr;
  CodecMic* mic_ = nullptr;
  uint32_t session_ = 0;  // the app task's: one more on every arm and disarm
  std::atomic<bool> test_{false};
  std::atomic<float> last_peak_{0};
  bool failed_ = false;
};

Detector* Detector::instance = nullptr;

int cmd_wake_test(int, char**) {
  if (!Detector::instance) {
    std::printf("@error wake-test: no wake word detector on this board (see the boot log)\n");
    return 0;
  }
  if (!Detector::instance->request_test()) {
    std::printf("@error wake-test: busy, the last one has not printed yet\n");
    return 0;
  }
  std::printf("wake-test: queued; the result follows within a second\n");
  return 0;
}

}  // namespace

hg::WakeWord* begin(CodecMic& mic) {
  auto* d = new (std::nothrow) Detector();
  if (!d || !d->begin(mic)) {
    ESP_LOGE(TAG, "wake word off: the detector did not start");
    delete d;  // frees what begin() got; nothing was published
    return nullptr;
  }
  return d;
}

void report(hg::json::Value& r, const hg::WakeStats& stats) {
  if (!Detector::instance) return;
  hg::json::Value wake = hg::json::Value::object();
  wake.set("armed", stats.armed).set("detections", static_cast<unsigned>(stats.detections))
      .set("no_speech", static_cast<unsigned>(stats.no_speech))
      .set("gain_db", static_cast<double>(hg::es7210_gain_step(stats.gain_db)))  // what the ES7210 gets
      .set("floor", static_cast<double>(stats.floor));
  if (const auto& t = stats.last_turn) {
    wake.set("last_turn", hg::json::Value::object()
                              .set("speech", t->speech)
                              .set("speech_ms", static_cast<unsigned>(t->speech_ms))
                              .set("rms_mean", std::round(static_cast<double>(t->rms_mean)))
                              .set("rms_max", std::round(static_cast<double>(t->rms_max)))
                              .set("peak", t->peak));
  }
  Detector::instance->report(wake);
  r.set("wake", wake);
}

void register_console() {
  esp_console_cmd_t cmd = {};
  cmd.command = "wake-test";
  cmd.help = "Run the wake word model on its built-in clip (or a second of silence) and print what it saw";
  cmd.func = &cmd_wake_test;
  esp_console_cmd_register(&cmd);
}

}  // namespace hgp::wake

#endif  // CONFIG_HG_WAKE_WORD
