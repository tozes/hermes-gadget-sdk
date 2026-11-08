#include "hg/vad.hpp"

#include <cmath>

namespace hg {

float rms(const int16_t* samples, size_t count) {
  if (count == 0) return 0.0f;
  double acc = 0;
  for (size_t i = 0; i < count; ++i) acc += double(samples[i]) * samples[i];
  return static_cast<float>(std::sqrt(acc / double(count)));
}

uint8_t level_percent(float r) {
  // ~-60 dBFS -> 0, 0 dBFS -> 100.
  if (r < 1.0f) return 0;
  float db = 20.0f * std::log10(r / 32768.0f);
  float pct = (db + 60.0f) * (100.0f / 60.0f);
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  return static_cast<uint8_t>(pct);
}

void Vad::reset(const Config& cfg) {
  cfg_ = cfg;
  noise_ = 0;
  elapsed_ms_ = speech_ms_ = silence_ms_ = 0;
  speech_blocks_ = 0;
  speech_rms_sum_ = 0;
  rms_max_ = 0;
  peak_ = 0;
}

VadLevel Vad::level() const {
  VadLevel l;
  l.speech = speech();
  l.speech_ms = speech_ms_;
  l.rms_mean = speech_blocks_ ? static_cast<float>(speech_rms_sum_ / speech_blocks_) : 0.0f;
  l.rms_max = rms_max_;
  l.peak = peak_;
  return l;
}

Vad::Result Vad::feed(const int16_t* samples, size_t count) {
  if (count == 0 || cfg_.sample_rate == 0) return Result::Continue;
  uint32_t ms = static_cast<uint32_t>(count * 1000 / cfg_.sample_rate);
  elapsed_ms_ += ms;
  if (elapsed_ms_ <= cfg_.ignore_ms) return Result::Continue;  // no_speech_ms is longer
  float r = rms(samples, count);
  if (r > rms_max_) rms_max_ = r;
  for (size_t i = 0; i < count; ++i) {
    const int a = samples[i] < 0 ? -int(samples[i]) : int(samples[i]);
    if (a > peak_) peak_ = a;
  }
  // Track the noise floor quickly downwards, slowly upwards. It starts at the
  // quiet end so speech that begins with the first block is still detected.
  if (noise_ == 0) noise_ = cfg_.min_threshold / cfg_.noise_ratio;
  noise_ = r < noise_ ? noise_ * 0.7f + r * 0.3f : noise_ * 0.995f + r * 0.005f;

  float threshold = noise_ * cfg_.noise_ratio;
  if (threshold < cfg_.min_threshold) threshold = cfg_.min_threshold;
  if (r >= threshold) {
    speech_ms_ += ms;
    silence_ms_ = 0;
    ++speech_blocks_;
    speech_rms_sum_ += r;
  } else if (speech_ms_ > 0) {
    silence_ms_ += ms;
  }
  if (speech() && silence_ms_ >= cfg_.end_silence_ms) return Result::EndOfSpeech;
  if (!speech() && elapsed_ms_ >= cfg_.no_speech_ms) return Result::NoSpeech;
  return Result::Continue;
}

}  // namespace hg
