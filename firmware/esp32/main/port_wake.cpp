// The wake word detector (CONFIG_HG_WAKE_WORD). While the core arms it (hg::WakeWord), the
// codec microphone's mic task hands it every 24 kHz chunk through a stream buffer instead of
// posting them to the app (CodecMic::set_listening). Its own task resamples them to 16 kHz
// (hg::Resample3to2), runs TensorFlow Lite Micro's audio frontend (40 features per 30 ms
// window, 10 ms step) and the streaming microWakeWord model (one inference per 3 steps), and
// a mean of the last N probabilities above the model's cutoff (hg::WakeWindow) posts one
// EventType::Wake. The model, its phrase, cutoff, window and arena come from the sdkconfig
// (CONFIG_HG_WAKE_WORD_MODEL and its values, taken from the model's manifest).
#include "port.hpp"  // first: pulls in FreeRTOS.h ahead of task.h/queue.h

#include "sdkconfig.h"

#if CONFIG_HG_WAKE_WORD

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>

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
// Audio from the mic task, 24 kHz: 0.5 s, in PSRAM. Read in 20 ms pieces.
constexpr size_t kAudioBuffer = 24000;
constexpr size_t kReadSamples = 480;
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

    constexpr size_t arena = CONFIG_HG_WAKE_WORD_ARENA;
    arena_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, arena, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena_) {
      ESP_LOGE(TAG, "no PSRAM for a %u B arena", unsigned(arena));
      return false;
    }
    tflite::MicroAllocator* alloc = tflite::MicroAllocator::Create(arena_, arena);
    tflite::MicroResourceVariables* vars = tflite::MicroResourceVariables::Create(alloc, 20);
    interp_ = new (std::nothrow) tflite::MicroInterpreter(tflite::GetModel(hg_wake_model_start), resolver_, alloc, vars);
    if (!interp_ || interp_->AllocateTensors() != kTfLiteOk) {
      ESP_LOGE(TAG, "AllocateTensors failed (arena %u B)", unsigned(arena));
      return false;
    }
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
    ESP_LOGI(TAG, "%s: stride %d, arena %u of %u B (PSRAM)", CONFIG_HG_WAKE_WORD_MODEL, stride_,
             unsigned(interp_->arena_used_bytes()), unsigned(arena));
    return true;
  }

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
  ~Detector() override { heap_caps_free(storage_); }

  // Everything that can fail comes before the task starts; after that the detector is
  // published (instance, the mic's sink) and never freed. On false the caller deletes it.
  bool begin(CodecMic& mic) {
    if (!model_.begin()) return false;
    window_ = hg::WakeWindow(CONFIG_HG_WAKE_WORD_CUTOFF_PCT / 100.0f, CONFIG_HG_WAKE_WORD_WINDOW,
                             kRefractoryMs / model_.ms_per_inference());
    storage_ = static_cast<uint8_t*>(heap_caps_malloc(kAudioBuffer + 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    static StaticStreamBuffer_t control;
    audio_ = storage_ ? xStreamBufferCreateStatic(kAudioBuffer, 1, storage_, &control) : nullptr;
    if (!audio_ || xTaskCreate(&Detector::task, "hg-wake", kTaskStack, this, 3, nullptr) != pdPASS) {
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
  // every chunk from then on even if the mic task hasn't seen this yet. Audio already queued
  // here is dropped on re-arm; a detection from it posts nothing once disarmed, and the
  // core ignores one posted just before (App::on_wake checks that it is armed).
  void set_armed(bool armed) override {
    if (armed) restart_ = true;  // before the first chunk arrives
    armed_ = armed;
    mic_->set_listening(armed);
    ESP_LOGI(TAG, "%s", armed ? "armed" : "disarmed");
  }

  void report(hg::json::Value& wake) const {
    wake.set("last_peak", static_cast<double>(last_peak_.load()))
        .set("invoke_us_mean", static_cast<unsigned>(model_.invoke_us_mean_.load()))
        .set("invoke_us_max", static_cast<unsigned>(model_.invoke_us_max_.load()))
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
  // On the mic task: whatever doesn't fit is dropped (the detector fell 0.5 s behind).
  static void sink(const int16_t* pcm, size_t samples) {
    xStreamBufferSend(instance->audio_, pcm, samples * sizeof(int16_t), 0);
  }

  static void task(void* arg) {
    auto* self = static_cast<Detector*>(arg);
    int16_t in[kReadSamples];
    int16_t pcm[hg::Resample3to2::max_out(kReadSamples)];
    for (;;) {
      // A second without audio (disarmed) still comes round for `wake-test`.
      const size_t got = xStreamBufferReceive(self->audio_, in, sizeof(in), pdMS_TO_TICKS(1000));
      if (self->test_.load()) {
        self->run_test();
        self->test_ = false;  // printed: the next wake-test may queue
      }
      if (self->restart_.exchange(false)) {
        // (Re)armed, or after a test: start from silence, not from audio left over.
        xStreamBufferReset(self->audio_);
        self->resampler_.reset();
        self->model_.reset();
        self->window_.reset();
        continue;
      }
      if (got < sizeof(int16_t)) continue;
      const size_t n = self->resampler_.process(in, got / sizeof(int16_t), pcm);
      bool detected = false;
      const bool ok = self->model_.feed(pcm, n, [self, &detected](float p) {
        if (self->window_.push(p)) detected = true;
      });
      if (!ok && !self->failed_) ESP_LOGE(TAG, "inference failed");
      self->failed_ = !ok;
      if (!detected) continue;
      // The window has started over; the model starts from silence too.
      self->model_.reset();
      const float mean = self->window_.mean();
      self->last_peak_ = mean;
      if (!self->armed_) continue;  // disarmed while this audio waited
      ESP_LOGI(TAG, "detected p=%.2f", static_cast<double>(mean));
      events::post(EventType::Wake);
    }
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
    restart_ = true;
  }

  Model model_;
  hg::Resample3to2 resampler_;
  hg::WakeWindow window_{1, 1, 0};
  uint8_t* storage_ = nullptr;  // audio_'s, in PSRAM
  StreamBufferHandle_t audio_ = nullptr;
  CodecMic* mic_ = nullptr;
  std::atomic<bool> armed_{false}, restart_{false}, test_{false};
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
      .set("no_speech", static_cast<unsigned>(stats.no_speech));
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
