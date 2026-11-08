// When a streaming wake word model's output counts as a detection, apart from ESP-IDF so it
// can be tested on the host (firmware/esp32/main/port_wake.cpp feeds it). A microWakeWord
// model gives one probability per inference; a detection is the mean of the last `window`
// of them above the model's cutoff. After one, the window starts empty and the next
// `refractory` probabilities can't detect, so one utterance wakes once.
#pragma once

namespace hg {

class WakeWindow {
 public:
  static constexpr int kMaxWindow = 10;

  // `window` is clamped to 1..kMaxWindow.
  WakeWindow(float cutoff, int window, int refractory)
      : cutoff_(cutoff),
        window_(window < 1 ? 1 : window > kMaxWindow ? kMaxWindow : window),
        refractory_(refractory) {}

  // Forgets the probabilities and any refractory period (the detector was re-armed).
  void reset() {
    filled_ = 0;
    holdoff_ = 0;
    mean_ = 0;
  }

  // One more probability; true when it completes a detection.
  bool push(float p) {
    if (holdoff_ > 0) {
      --holdoff_;
      return false;
    }
    probs_[next_] = p;
    next_ = next_ + 1 == window_ ? 0 : next_ + 1;
    if (filled_ < window_ && ++filled_ < window_) return false;
    float sum = 0;
    for (int i = 0; i < window_; ++i) sum += probs_[i];
    mean_ = sum / window_;
    if (mean_ <= cutoff_) return false;
    filled_ = 0;
    holdoff_ = refractory_;
    return true;
  }

  // The window's mean after the last full push(): at a detection, the mean that crossed the cutoff.
  float mean() const { return mean_; }
  int window() const { return window_; }

 private:
  float cutoff_;
  int window_, refractory_;
  float probs_[kMaxWindow] = {};
  int next_ = 0, filled_ = 0, holdoff_ = 0;
  float mean_ = 0;
};

}  // namespace hg
