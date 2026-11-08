// When the microphone ADC is powered, apart from ESP-IDF so it can be tested on
// the host (firmware/esp32/main/port_codec.cpp drives it from the mic task).
//
// A codec that only records (an ES7210 next to a separate DAC) is switched off
// while nothing records or listens: it otherwise runs, and keeps the CPU busy
// reading it, around the clock. A codec shared with the speaker stays on, as before.
// Listening is the wake word detector's always-on capture (port_wake.cpp): the
// ADC stays on and chunks go to it instead of the app. A recording takes over.
#pragma once
#include <cstdint>

namespace hg {

class MicPower {
 public:
  // Chunks dropped after power-up while the ADC settles (20 ms each).
  static constexpr int kSettleChunks = 1;

  // The codec starts open (CodecAudio::begin opens it), so on.
  explicit MicPower(bool switchable) : switchable_(switchable) {}
  bool on() const { return on_; }

  // One turn of the mic task. Ops supplies:
  //   bool power_up()               start the ADC and restore its gain; false on failure
  //   void power_down()
  //   void drain()                  drop audio buffered while the ADC was off
  //   bool read(int16_t* chunk)     one chunk; false on a read error
  //   void deliver(const int16_t* chunk)   to the app (recording)
  //   void listen(const int16_t* chunk)    to the wake word detector
  //   void wait()                   block until a capture may be wanted
  //   void pause()                  back off after an error
  template <class Ops>
  void turn(bool capturing, Ops& ops, int16_t* chunk, bool listening = false) {
    if (switchable_ && !capturing && !listening) {
      if (on_) {
        ops.power_down();
        on_ = false;
      }
      ops.wait();
      return;
    }
    if (!on_) {
      if (!ops.power_up()) {
        ops.pause();
        return;
      }
      on_ = true;
      ops.drain();
      discard_ = kSettleChunks;
    }
    if (!ops.read(chunk)) {
      ops.pause();
      return;
    }
    if (discard_ > 0) {
      --discard_;
      return;
    }
    // A shared codec keeps reading while idle so a capture starts with fresh
    // samples; only a capture (or listening) is delivered.
    if (capturing) ops.deliver(chunk);
    else if (listening) ops.listen(chunk);
  }

 private:
  bool switchable_;
  bool on_ = true;
  int discard_ = 0;
};

}  // namespace hg
