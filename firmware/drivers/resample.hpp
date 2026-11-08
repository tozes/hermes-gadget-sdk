// 3:2 sample-rate conversion (24 kHz to 16 kHz), apart from ESP-IDF so it can be tested on
// the host. The wake word detector (firmware/esp32/main/port_wake.cpp) needs 16 kHz, and the
// codec boards run their microphones at 24 kHz only (hgp::CodecAudio::kRate).
//
// Conceptually: zero-stuff to 48 kHz (upsample by 2), low-pass, keep every third sample. As a
// polyphase FIR, each output is one dot product of kPhaseTaps input samples with the taps of
// one of two phases, so nothing runs at 48 kHz. The low-pass is a Kaiser-windowed sinc, cut
// off just under the new 8 kHz Nyquist rate (the wake word frontend stops at 7.5 kHz).
// Allocation-free; process() takes any chunk size and gives the same output as one block.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace hg {

class Resample3to2 {
 public:
  static constexpr int kPhaseTaps = 48;  // 96 taps at 48 kHz
  static constexpr double kCutoffHz = 7800, kKaiserBeta = 6.5;

  Resample3to2() {
    // h[k], k = 0..2*kPhaseTaps-1, at 48 kHz, scaled to a DC gain of 2 to make up for the
    // zero-stuffing. Output m uses phase p = m % 2 (taps h[2j + p]) against input
    // floor(3m / 2) - j; stored oldest-input-first to match the history window.
    constexpr int n = 2 * kPhaseTaps;
    constexpr double kPi = 3.14159265358979323846;
    double h[n], sum = 0;
    for (int k = 0; k < n; ++k) {
      const double t = k - (n - 1) / 2.0;
      const double x = 2 * kCutoffHz / 48000 * t;
      const double sinc = x == 0 ? 1 : std::sin(kPi * x) / (kPi * x);
      const double r = 2 * t / (n - 1);
      h[k] = sinc * bessel_i0(kKaiserBeta * std::sqrt(1 - r * r)) / bessel_i0(kKaiserBeta);
      sum += h[k];
    }
    for (int p = 0; p < 2; ++p)
      for (int j = 0; j < kPhaseTaps; ++j) taps_[p][kPhaseTaps - 1 - j] = static_cast<float>(h[2 * j + p] * 2 / sum);
    reset();
  }

  // Forgets the input history (silence before the next sample).
  void reset() {
    for (float& s : history_) s = 0;
    pos_ = 0;
    phase_ = 0;
  }

  // Output samples for `n` more input samples: at most ceil(2n / 3).
  static constexpr size_t max_out(size_t n) { return (2 * n + 2) / 3; }

  // Converts `n` input samples into `out` (room for max_out(n)); returns how many it wrote.
  size_t process(const int16_t* in, size_t n, int16_t* out) {
    size_t written = 0;
    for (size_t i = 0; i < n; ++i) {
      // Each sample goes in twice, so the last kPhaseTaps samples are always contiguous.
      history_[pos_] = history_[pos_ + kPhaseTaps] = in[i];
      pos_ = pos_ + 1 == kPhaseTaps ? 0 : pos_ + 1;
      // Of every three inputs, the first ends an even output and the second an odd one.
      const int phase = phase_;
      phase_ = phase_ == 2 ? 0 : phase_ + 1;
      if (phase == 2) continue;
      const float* window = history_ + pos_;  // oldest first
      const float* taps = taps_[phase];
      float acc = 0;
      for (int j = 0; j < kPhaseTaps; ++j) acc += taps[j] * window[j];
      const float rounded = std::nearbyint(acc);
      out[written++] = static_cast<int16_t>(rounded > 32767 ? 32767 : rounded < -32768 ? -32768 : rounded);
    }
    return written;
  }

 private:
  // The zeroth-order modified Bessel function of the first kind (the Kaiser window), by its series.
  static double bessel_i0(double x) {
    double term = 1, sum = 1;
    for (int k = 1; k < 32; ++k) {
      term *= (x / (2 * k)) * (x / (2 * k));
      sum += term;
    }
    return sum;
  }

  float taps_[2][kPhaseTaps];
  float history_[2 * kPhaseTaps];
  int pos_ = 0;    // where the next sample goes; history_[pos_..pos_ + kPhaseTaps) is the window
  int phase_ = 0;  // input index mod 3
};

}  // namespace hg
