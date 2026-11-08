#include "check.hpp"
#include "wake_window.hpp"

namespace {

// Pushes `p` n times; returns how many pushes detected.
int push(hg::WakeWindow& w, float p, int n) {
  int hits = 0;
  for (int i = 0; i < n; ++i) hits += w.push(p);
  return hits;
}

}  // namespace

TEST("wake window: a detection needs a full window whose mean is above the cutoff") {
  hg::WakeWindow w(0.5f, 3, 10);
  CHECK(!w.push(0.9f));
  CHECK(!w.push(0.9f));
  CHECK(w.push(0.9f));  // the third fills the window
  hg::WakeWindow at_cutoff(0.5f, 2, 0);
  CHECK_EQ(push(at_cutoff, 0.5f, 5), 0);  // equal is not above
}

TEST("wake window: it is the mean that counts, not one probability") {
  hg::WakeWindow w(0.6f, 4, 0);
  // One spike among low values stays under; a sustained rise crosses.
  CHECK_EQ(push(w, 0.1f, 3), 0);
  CHECK(!w.push(1.0f));  // mean 0.325
  CHECK(!w.push(0.8f));  // mean 0.5
  CHECK(w.push(0.8f));   // mean 0.675
  CHECK(w.mean() > 0.67f && w.mean() < 0.68f);
}

TEST("wake window: one utterance wakes once; the refractory period then ends") {
  hg::WakeWindow w(0.5f, 2, 5);
  CHECK_EQ(push(w, 0.9f, 2), 1);
  // Five probabilities ignored, then the window refills from empty (two more) before the next.
  CHECK_EQ(push(w, 0.9f, 6), 0);
  CHECK(w.push(0.9f));
}

TEST("wake window: reset() forgets the window and the refractory period") {
  hg::WakeWindow w(0.5f, 3, 100);
  push(w, 0.9f, 2);
  w.reset();
  CHECK(!w.push(0.9f));  // the two before the reset no longer count
  CHECK(!w.push(0.9f));
  CHECK(w.push(0.9f));
  w.reset();  // straight after a detection: no refractory left
  CHECK_EQ(push(w, 0.9f, 3), 1);
}

TEST("wake window: the window length is clamped to what it can hold") {
  CHECK_EQ(hg::WakeWindow(0.5f, 50, 0).window(), hg::WakeWindow::kMaxWindow);
  CHECK_EQ(hg::WakeWindow(0.5f, 0, 0).window(), 1);
}
