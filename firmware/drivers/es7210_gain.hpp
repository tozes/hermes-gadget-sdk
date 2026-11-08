// The ES7210 microphone gain the codec can be given, apart from ESP-IDF so it can be tested on
// the host (firmware/esp32/main/port_codec.cpp uses it for the wake word's gain).
//
// The ES7210's PGA has 0..33 dB in 3 dB steps, then 34.5, 36 and 37.5 dB. esp_codec_dev's
// es7210 driver (get_db) maps a request to the nearest step except 33, which it sets as 30
// (its 33-34.5 branch returns GAIN_30DB), so 33 dB is not reachable through it.
#pragma once

namespace hg {

// The highest step esp_codec_dev_set_in_gain reaches at or below `db` (0 below 0): rounded down,
// so a step never adds more gain, and clipping, than was asked for.
inline float es7210_gain_step(float db) {
  if (db >= 37.5f) return 37.5f;
  if (db >= 36.0f) return 36.0f;
  if (db >= 34.5f) return 34.5f;
  if (db >= 30.0f) return 30.0f;  // 33 sets 30
  if (!(db > 0.0f)) return 0.0f;  // and NaN
  return static_cast<float>(static_cast<int>(db / 3.0f) * 3);
}

}  // namespace hg
