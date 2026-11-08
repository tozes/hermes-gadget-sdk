// Energy-based voice activity detection for hands-free (tap) talk mode.
//
// Deliberately simple: an adaptive noise floor and a speech threshold above
// it. Good enough to end an utterance after a pause; boards with a DSP wake
// word / VAD engine can replace it behind the same interface later.
#pragma once

#include <cstddef>
#include <cstdint>

namespace hg {

// RMS of a PCM16 block in [0, 32768].
float rms(const int16_t* samples, size_t count);
// Maps RMS to a perceptual 0..100 meter value (log scale).
uint8_t level_percent(float rms_value);

// What a recording's VAD heard after its ignored start (Vad::Config::ignore_ms), for tuning the
// floor and the microphone gain: whether it counted as speech (Vad::speech), how long its speech
// blocks lasted, their mean RMS (0 without any), the loudest block's RMS and the largest sample.
struct VadLevel {
  bool speech = false;
  uint32_t speech_ms = 0;
  float rms_mean = 0;
  float rms_max = 0;
  int peak = 0;
};

class Vad {
 public:
  struct Config {
    uint32_t sample_rate = 16000;
    uint32_t end_silence_ms = 900;   // silence after speech that ends the utterance
    uint32_t no_speech_ms = 6000;    // give up if nobody speaks at all
    float min_threshold = 500.0f;    // absolute RMS floor for "speech"
    float noise_ratio = 3.0f;        // speech must exceed noise floor * ratio
    // Blocks that end within this long of reset() are neither speech nor noise, nor in level();
    // they still count towards no_speech_ms. For a microphone's start-up pop.
    uint32_t ignore_ms = 0;
  };

  enum class Result : uint8_t { Continue, EndOfSpeech, NoSpeech };

  void reset(const Config& cfg);
  Result feed(const int16_t* samples, size_t count);
  bool heard_speech() const { return speech_ms_ > 0; }
  // Enough speech for an utterance: what EndOfSpeech needs, and what NoSpeech lacks.
  bool speech() const { return speech_ms_ >= kMinSpeechMs; }
  VadLevel level() const;

 private:
  // A little sustained speech before an end-of-speech can trigger, so a single click
  // does not count as an utterance.
  static constexpr uint32_t kMinSpeechMs = 200;
  Config cfg_;
  float noise_ = 0;
  uint32_t elapsed_ms_ = 0;
  uint32_t speech_ms_ = 0;
  uint32_t silence_ms_ = 0;
  // For level(): over the blocks after ignore_ms.
  uint32_t speech_blocks_ = 0;
  double speech_rms_sum_ = 0;
  float rms_max_ = 0;
  int peak_ = 0;
};

}  // namespace hg
