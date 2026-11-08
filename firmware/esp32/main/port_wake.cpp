// Wake-word detector (CONFIG_HG_WAKE_WORD, spike): 16 kHz PCM -> TFLM micro frontend
// (40 features per 30 ms window, 10 ms step) -> streaming microWakeWord model ->
// mean of the last N probabilities against the model's cutoff. Nothing feeds it the
// microphone yet; the console's `wake-test` runs it on a built-in clip.
#include "port.hpp"  // first: pulls in FreeRTOS.h ahead of task.h/queue.h

#include "sdkconfig.h"

#if CONFIG_HG_WAKE_WORD

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "esp_console.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "frontend.h"
#include "frontend_util.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace hgp::wake {
namespace {

const char* TAG = "hg.wake";

constexpr int kSampleRate = 16000;
constexpr size_t kChunk = kSampleRate / 100;  // 10 ms, one feature step
constexpr int kFeatures = 40;
// The frontend's uint16 outputs are scaled by this before quantising: microWakeWord's
// training features are the same outputs as floats (pymicro-features does the same).
constexpr float kFeatureScale = 0.0390625f;
constexpr int kMaxWindow = 10;

// One model and its manifest (micro.* in the model's JSON). The arena is the
// manifest's tensor_arena_size, which also holds the streaming state variables.
struct ModelSpec {
  const char* name;
  const uint8_t* data;
  const int16_t* clip;
  size_t clip_samples;
  float cutoff;
  int window;
  size_t arena;
};

// Puts wake/<file> in flash as <sym>_start .. <sym>_end, 16-byte aligned as TFLM
// wants its model. HG_WAKE_DIR comes from CMakeLists.txt.
#define HG_INCBIN(sym, file)                                                              \
  asm(".section .rodata." #sym ",\"a\"\n.balign 16\n.global " #sym "_start\n" #sym       \
      "_start:\n.incbin \"" HG_WAKE_DIR "/" file "\"\n.global " #sym "_end\n" #sym "_end:\n" \
      ".previous\n");                                                                     \
  extern const uint8_t sym##_start[] asm(#sym "_start");                                  \
  extern const uint8_t sym##_end[] asm(#sym "_end");

#if CONFIG_HG_WAKE_WORD_HEY_HERMES
HG_INCBIN(hey_hermes_tflite, "hey_hermes.tflite")
#if CONFIG_HG_WAKE_WORD_TEST_CLIPS
HG_INCBIN(hey_hermes_clip, "hey_hermes_test.pcm")
#endif
#endif
#if CONFIG_HG_WAKE_WORD_HEY_JARVIS
HG_INCBIN(hey_jarvis_tflite, "hey_jarvis.tflite")
#if CONFIG_HG_WAKE_WORD_TEST_CLIPS
HG_INCBIN(hey_jarvis_clip, "hey_jarvis_test.pcm")
#endif
#endif

#if CONFIG_HG_WAKE_WORD_TEST_CLIPS
#define HG_CLIP(name) reinterpret_cast<const int16_t*>(name##_clip_start), \
                      size_t(name##_clip_end - name##_clip_start) / 2
#else
#define HG_CLIP(name) nullptr, 0
#endif

// hey-hermes.json (miketomkins/hey-hermes-wakeword-model) and hey_jarvis.json
// (esphome/micro-wake-word-models models/v2).
const ModelSpec kModels[] = {
#if CONFIG_HG_WAKE_WORD_HEY_HERMES
    {"hermes", hey_hermes_tflite_start, HG_CLIP(hey_hermes), 0.32f, 5, 65536},
#endif
#if CONFIG_HG_WAKE_WORD_HEY_JARVIS
    {"jarvis", hey_jarvis_tflite_start, HG_CLIP(hey_jarvis), 0.97f, 5, 22860},
#endif
};

// The operators both models use (checked with the LiteRT interpreter on the host).
using Resolver = tflite::MicroMutableOpResolver<13>;

class Detector {
 public:
  // Allocates everything once: the arena in PSRAM, the interpreter and the frontend.
  bool begin(const ModelSpec& spec) {
    spec_ = &spec;
    FrontendConfig fc;
    FrontendFillConfigWithDefaults(&fc);
    fc.window.size_ms = 30;
    fc.window.step_size_ms = 10;
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
      ESP_LOGE(TAG, "%s: frontend setup failed", spec.name);
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

    arena_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, spec.arena, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena_) {
      ESP_LOGE(TAG, "%s: no PSRAM for a %u B arena", spec.name, unsigned(spec.arena));
      return false;
    }
    const tflite::Model* model = tflite::GetModel(spec.data);
    tflite::MicroAllocator* alloc = tflite::MicroAllocator::Create(arena_, spec.arena);
    tflite::MicroResourceVariables* vars = tflite::MicroResourceVariables::Create(alloc, 20);
    interp_ = new (std::nothrow) tflite::MicroInterpreter(model, resolver_, alloc, vars);
    if (!interp_ || interp_->AllocateTensors() != kTfLiteOk) {
      ESP_LOGE(TAG, "%s: AllocateTensors failed (arena %u B)", spec.name, unsigned(spec.arena));
      return false;
    }
    TfLiteTensor* in = interp_->input(0);
    TfLiteTensor* out = interp_->output(0);
    if (in->type != kTfLiteInt8 || in->dims->size != 3 || in->dims->data[2] != kFeatures ||
        out->type != kTfLiteUInt8 || spec.window > kMaxWindow) {
      ESP_LOGE(TAG, "%s: unexpected model tensors", spec.name);
      return false;
    }
    stride_ = in->dims->data[1];
    in_scale_ = in->params.scale;
    in_zero_ = in->params.zero_point;
    out_scale_ = out->params.scale;
    out_zero_ = out->params.zero_point;
    ESP_LOGI(TAG, "%s: stride %d, arena %u of %u B", spec.name, stride_, unsigned(interp_->arena_used_bytes()),
             unsigned(spec.arena));
    return true;
  }

  // Clears the audio and model state and the statistics (not on the audio path).
  void reset() {
    FrontendReset(&frontend_);
    interp_->Reset();
    frames_ = filled_ = 0;
    detected_ = false;
    max_p_ = max_mean_ = 0;
    invokes_ = invoke_us_total_ = invoke_us_max_ = detect_invoke_ = 0;
  }

  // Feeds 16 kHz samples. Allocation-free. Returns true once the window mean first
  // crosses the cutoff since reset().
  bool feed(const int16_t* pcm, size_t n) {
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
      if (interp_->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "%s: Invoke failed", spec_->name);
        return false;
      }
      const uint32_t us = uint32_t(esp_timer_get_time() - t0);
      ++invokes_;
      invoke_us_total_ += us;
      if (us > invoke_us_max_) invoke_us_max_ = us;
      const float p = (float(interp_->output(0)->data.uint8[0]) - out_zero_) * out_scale_;
      if (p > max_p_) max_p_ = p;
      window_[filled_ % spec_->window] = p;
      if (++filled_ < spec_->window) continue;
      float sum = 0;
      for (int i = 0; i < spec_->window; ++i) sum += window_[i];
      const float mean = sum / spec_->window;
      if (mean > max_mean_) max_mean_ = mean;
      if (!detected_ && mean > spec_->cutoff) {
        detected_ = true;
        detect_invoke_ = invokes_;
        return true;
      }
    }
    return false;
  }

  const ModelSpec* spec_ = nullptr;
  float max_p_ = 0, max_mean_ = 0;
  uint32_t invokes_ = 0, invoke_us_total_ = 0, invoke_us_max_ = 0, detect_invoke_ = 0;
  int stride_ = 0;

 private:
  FrontendState frontend_{};
  Resolver resolver_;
  uint8_t* arena_ = nullptr;
  tflite::MicroInterpreter* interp_ = nullptr;
  float in_scale_ = 1, out_scale_ = 1;
  int in_zero_ = 0, out_zero_ = 0;
  int frames_ = 0, filled_ = 0;
  bool detected_ = false;
  float window_[kMaxWindow] = {};
};

constexpr size_t kModelCount = sizeof(kModels) / sizeof(kModels[0]);
Detector* g_detectors[kModelCount] = {};

void run_test(void* arg) {
  const size_t m = reinterpret_cast<size_t>(arg);
  const ModelSpec& spec = kModels[m];
  Detector*& d = g_detectors[m];
  if (!d) {
    d = new (std::nothrow) Detector();
    if (!d || !d->begin(spec)) {
      std::printf("@error wake: %s did not start (see log)\n", spec.name);
      vTaskDelete(nullptr);
      return;
    }
  }
  d->reset();
  static const int16_t kSilence[kChunk] = {};
  const size_t samples = spec.clip ? spec.clip_samples : size_t(kSampleRate);
  for (size_t off = 0; off < samples; off += kChunk) {
    const size_t n = samples - off < kChunk ? samples - off : kChunk;
    d->feed(spec.clip ? spec.clip + off : kSilence, n);
  }
  std::printf("wake-test %s on %s (%u ms): max p %.3f, max mean %.3f, cutoff %.2f, %s at %u ms; "
              "%u invokes, %u us mean, %u us max\n",
              spec.name, spec.clip ? "clip" : "silence", unsigned(samples * 1000 / kSampleRate), d->max_p_,
              d->max_mean_, spec.cutoff, d->detect_invoke_ ? "DETECTED" : "not detected",
              unsigned(d->detect_invoke_ * d->stride_ * 10), unsigned(d->invokes_),
              unsigned(d->invokes_ ? d->invoke_us_total_ / d->invokes_ : 0), unsigned(d->invoke_us_max_));
  vTaskDelete(nullptr);
}

int cmd_wake_test(int argc, char** argv) {
  size_t m = 0;
  if (argc > 1) {
    for (m = 0; m < kModelCount && std::strcmp(kModels[m].name, argv[1]) != 0; ++m) {
    }
  }
  if (m >= kModelCount) {
    std::printf("@error wake-test: models built in:");
    for (const ModelSpec& s : kModels) std::printf(" %s", s.name);
    std::printf("\n");
    return 0;
  }
  // Its own task: the REPL task's stack is too small for the interpreter.
  if (xTaskCreate(run_test, "wake_test", 8192, reinterpret_cast<void*>(m), 5, nullptr) != pdPASS)
    std::printf("@error wake-test: no memory for the task\n");
  return 0;
}

}  // namespace

void register_console() {
  static_assert(kModelCount > 0, "CONFIG_HG_WAKE_WORD needs at least one model");
  esp_console_cmd_t cmd = {};
  cmd.command = "wake-test";
  cmd.help = "Run a wake-word model on its built-in clip: wake-test [hermes|jarvis]";
  cmd.func = &cmd_wake_test;
  esp_console_cmd_register(&cmd);
}

}  // namespace hgp::wake

#endif  // CONFIG_HG_WAKE_WORD
