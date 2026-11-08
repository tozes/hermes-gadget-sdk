#include "check.hpp"
#include "es7210_gain.hpp"

TEST("es7210 gain: rounds down to a step the driver sets; 33 dB is not one, so it gives 30") {
  CHECK_EQ(hg::es7210_gain_step(24.0f), 24.0f);  // the 1.75C's own
  CHECK_EQ(hg::es7210_gain_step(29.9f), 27.0f);
  CHECK_EQ(hg::es7210_gain_step(30.0f), 30.0f);
  CHECK_EQ(hg::es7210_gain_step(33.0f), 30.0f);
  CHECK_EQ(hg::es7210_gain_step(34.5f), 34.5f);
  CHECK_EQ(hg::es7210_gain_step(35.9f), 34.5f);
  CHECK_EQ(hg::es7210_gain_step(36.0f), 36.0f);
  CHECK_EQ(hg::es7210_gain_step(37.5f), 37.5f);
  CHECK_EQ(hg::es7210_gain_step(40.0f), 37.5f);
  CHECK_EQ(hg::es7210_gain_step(2.9f), 0.0f);
  CHECK_EQ(hg::es7210_gain_step(-3.0f), 0.0f);
}
