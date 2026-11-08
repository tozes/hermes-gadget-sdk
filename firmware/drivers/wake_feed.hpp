// The microphone's hand-off to the wake word detector, apart from ESP-IDF so it can be tested
// on the host (firmware/esp32/main/port_wake.cpp: the mic task pushes, the detector's task
// takes). A single-producer, single-consumer ring of whole chunks, each tagged with the armed
// session (hg::WakeWord::set_armed) it was heard in and its place in the mic's sequence.
//
// - A chunk goes in whole or not at all: on a full ring it is dropped and counted, never split.
// - Only push() moves the head and only take() the tail, and nothing resets either, so
//   re-arming never races the producer: take() skips chunks from an earlier session instead.
// - take() says when a chunk doesn't follow the last one it handed out (a new session, or
//   chunks dropped in between), so the detector starts its audio state over there.
//
// The ring itself is the caller's (the port's is in PSRAM); the cursors here are written by one
// task each, with no read-modify-write, so they may live anywhere.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hg {

template <size_t kChunk>
class WakeFeed {
 public:
  struct Slot {
    uint32_t session;
    uint32_t seq;
    int16_t pcm[kChunk];
  };

  // `count` slots at `ring` (null: nothing fits); one always stays empty.
  WakeFeed(Slot* ring, size_t count) : ring_(ring), count_(ring ? count : 0) {}

  // The armed session the producer tags chunks with; 0 while disarmed, when push() takes nothing.
  void set_session(uint32_t session) { session_.store(session, std::memory_order_release); }
  // `session` is armed now: a detection from its audio may be published.
  bool current(uint32_t session) const { return session && session == session_.load(std::memory_order_acquire); }

  // Producer: one chunk of kChunk samples. False when disarmed or the ring is full (dropped).
  bool push(const int16_t* pcm) {
    const uint32_t seq = seq_++;  // counts dropped chunks too, so the consumer sees the gap
    const uint32_t session = session_.load(std::memory_order_acquire);
    if (!session || count_ < 2) return false;
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next = head + 1 == count_ ? 0 : head + 1;
    if (next == tail_.load(std::memory_order_acquire)) {
      dropped_.store(dropped_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
      return false;
    }
    Slot& slot = ring_[head];
    slot.session = session;
    slot.seq = seq;
    std::memcpy(slot.pcm, pcm, sizeof(slot.pcm));
    head_.store(next, std::memory_order_release);
    return true;
  }

  // Consumer: hands the oldest chunk of the session armed now to
  // f(const int16_t* pcm, size_t samples, uint32_t session, bool restart), skipping any from
  // earlier sessions before it. `restart`: start the audio state over before this chunk.
  // False when no chunk of the current session is waiting.
  template <class F>
  bool take(F&& f) {
    for (;;) {
      const size_t tail = tail_.load(std::memory_order_relaxed);
      if (tail == head_.load(std::memory_order_acquire)) return false;
      const Slot& slot = ring_[tail];
      const bool current = slot.session == session_.load(std::memory_order_acquire);
      if (current) {
        const bool restart = slot.session != last_session_ || slot.seq != last_seq_ + 1;
        last_session_ = slot.session;
        last_seq_ = slot.seq;
        f(static_cast<const int16_t*>(slot.pcm), kChunk, slot.session, restart);
      }
      tail_.store(tail + 1 == count_ ? 0 : tail + 1, std::memory_order_release);  // the slot is the producer's again
      if (current) return true;
    }
  }

  // Consumer: the next chunk taken restarts even if it follows on (the model was used meanwhile).
  void restart() { last_session_ = 0; }

  // Chunks dropped on a full ring since boot.
  uint32_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  Slot* ring_;
  size_t count_;
  std::atomic<uint32_t> session_{0};
  std::atomic<size_t> head_{0}, tail_{0};
  std::atomic<uint32_t> dropped_{0};
  uint32_t seq_ = 0;                             // producer's
  uint32_t last_session_ = 0, last_seq_ = 0;     // consumer's
};

}  // namespace hg
