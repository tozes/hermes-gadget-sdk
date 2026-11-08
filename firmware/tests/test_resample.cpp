#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "resample.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kAmplitude = 10000;

// One second of a 24 kHz tone, converted to 16 kHz in one block.
std::vector<int16_t> tone_at_16k(double hz) {
  std::vector<int16_t> in(24000);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<int16_t>(std::lround(kAmplitude * std::sin(2 * kPi * hz * i / 24000)));
  hg::Resample3to2 r;
  std::vector<int16_t> out(hg::Resample3to2::max_out(in.size()));
  out.resize(r.process(in.data(), in.size(), out.data()));
  return out;
}

// Amplitude of the `hz` component of 16 kHz samples, after the filter has settled (first 960
// samples skipped; 15,000 left, a whole number of periods of 1 kHz).
double amplitude_at(const std::vector<int16_t>& pcm, double hz) {
  double re = 0, im = 0;
  const size_t from = 960, n = 15000;
  for (size_t i = 0; i < n; ++i) {
    re += pcm[from + i] * std::cos(2 * kPi * hz * i / 16000);
    im += pcm[from + i] * std::sin(2 * kPi * hz * i / 16000);
  }
  return 2 * std::sqrt(re * re + im * im) / n;
}

double peak_amplitude(const std::vector<int16_t>& pcm) {
  double rms = 0;
  for (size_t i = 960; i < pcm.size(); ++i) rms += double(pcm[i]) * pcm[i];
  return std::sqrt(2 * rms / (pcm.size() - 960));
}

}  // namespace

TEST("resample 3:2: three inputs give two outputs") {
  CHECK_EQ(tone_at_16k(1000).size(), size_t(16000));
  CHECK_EQ(hg::Resample3to2::max_out(1), size_t(1));
  CHECK_EQ(hg::Resample3to2::max_out(320), size_t(214));
}

TEST("resample 3:2: a 1 kHz tone comes out at 1 kHz with its amplitude") {
  const std::vector<int16_t> out = tone_at_16k(1000);
  // All the energy is at 1 kHz (a tone at the wrong rate would miss this bin), within 0.1 dB.
  CHECK(std::fabs(amplitude_at(out, 1000) - kAmplitude) < kAmplitude * 0.012);
  CHECK(std::fabs(peak_amplitude(out) - kAmplitude) < kAmplitude * 0.012);
  // A 6 kHz tone, near the top of the wake word frontend's band, passes within 0.1 dB too.
  CHECK(std::fabs(amplitude_at(tone_at_16k(6000), 6000) - kAmplitude) < kAmplitude * 0.012);
}

TEST("resample 3:2: tones above the new 8 kHz Nyquist rate are attenuated by 60 dB or more") {
  // Above 8 kHz they would alias into the band the detector hears; 10 kHz would fold to 6 kHz.
  for (double hz : {9000.0, 10000.0, 11000.0}) CHECK(peak_amplitude(tone_at_16k(hz)) < kAmplitude / 1000);
}

TEST("resample 3:2: chunk boundaries don't change the output") {
  std::vector<int16_t> in(4801);
  for (size_t i = 0; i < in.size(); ++i)
    in[i] = static_cast<int16_t>(std::lround(12000 * std::sin(2 * kPi * 440 * i / 24000) + 3000 * std::sin(2 * kPi * 3100 * i / 24000)));
  hg::Resample3to2 whole;
  std::vector<int16_t> expected(hg::Resample3to2::max_out(in.size()));
  expected.resize(whole.process(in.data(), in.size(), expected.data()));

  hg::Resample3to2 chunked;
  std::vector<int16_t> got;
  const size_t sizes[] = {1, 2, 5, 7, 320, 13, 1, 319, 3, 641};
  size_t at = 0;
  for (size_t k = 0; at < in.size(); ++k) {
    const size_t n = std::min(sizes[k % (sizeof(sizes) / sizeof(sizes[0]))], in.size() - at);
    int16_t out[hg::Resample3to2::max_out(641)];
    const size_t w = chunked.process(in.data() + at, n, out);
    CHECK(w <= hg::Resample3to2::max_out(n));
    got.insert(got.end(), out, out + w);
    at += n;
  }
  CHECK(got == expected);
}

TEST("resample 3:2: reset() forgets the input history") {
  std::vector<int16_t> loud(300, 20000), quiet(30, 0), out(hg::Resample3to2::max_out(300));
  hg::Resample3to2 r;
  r.process(loud.data(), loud.size(), out.data());
  r.reset();
  const size_t w = r.process(quiet.data(), quiet.size(), out.data());
  bool silent = true;
  for (size_t i = 0; i < w; ++i) silent = silent && out[i] == 0;
  CHECK(silent);
}

TEST("resample 3:2: full-scale input clips instead of wrapping") {
  // A square wave overshoots after the low-pass (Gibbs); the overshoot must clamp, not wrap.
  std::vector<int16_t> in(2400), out(hg::Resample3to2::max_out(2400));
  for (size_t i = 0; i < in.size(); ++i) in[i] = (i / 40) % 2 ? -32768 : 32767;
  hg::Resample3to2 r;
  const size_t w = r.process(in.data(), in.size(), out.data());
  bool sign_follows = true;
  for (size_t i = 200; i < w; ++i) {
    // The input sample at the middle of the filter: 47.5 samples back at 48 kHz.
    const size_t src = i * 3 / 2 - 24;
    // Plateau samples, including the overshoot near the edges, keep the input's sign.
    if (src % 40 >= 4 && src % 40 < 36) sign_follows = sign_follows && ((src / 40) % 2 ? out[i] < 0 : out[i] > 0);
  }
  CHECK(sign_follows);
}
