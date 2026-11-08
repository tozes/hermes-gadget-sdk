#include <vector>

#include "check.hpp"
#include "wake_feed.hpp"

namespace {

constexpr size_t kChunk = 4;
using Feed = hg::WakeFeed<kChunk>;

// A ring of `slots` (one stays empty) and what take() handed over.
struct Rig {
  std::vector<Feed::Slot> ring;
  Feed feed;
  explicit Rig(size_t slots) : ring(slots), feed(ring.data(), ring.size()) {}

  struct Taken {
    int16_t first = 0, last = 0;
    bool whole = false;  // every sample of the chunk is `first`
    uint32_t session = 0;
    bool restart = false;
  };
  bool push(int16_t value) {
    int16_t pcm[kChunk];
    for (int16_t& s : pcm) s = value;
    return feed.push(pcm);
  }
  bool take(Taken& t) {
    return feed.take([&](const int16_t* pcm, size_t n, uint32_t session, bool restart) {
      t = {pcm[0], pcm[n - 1], n == kChunk, session, restart};
      for (size_t i = 0; i < n; ++i) t.whole = t.whole && pcm[i] == pcm[0];
    });
  }
};

}  // namespace

TEST("wake feed: nothing goes in while disarmed") {
  Rig r(4);
  CHECK(!r.push(1));
  Rig::Taken t;
  CHECK(!r.take(t));
  CHECK_EQ(r.feed.dropped(), uint32_t(0));  // not a loss: nobody was listening
}

TEST("wake feed: a full ring drops whole chunks, counts them, and the next one restarts the audio") {
  Rig r(4);
  r.feed.set_session(1);
  CHECK(r.push(0x1234));
  CHECK(r.push(0x1235));
  CHECK(r.push(0x1236));
  CHECK(!r.push(0x1237));  // full: one slot stays empty
  CHECK_EQ(r.feed.dropped(), uint32_t(1));
  Rig::Taken t;
  for (int16_t want : {0x1234, 0x1235, 0x1236}) {
    CHECK(r.take(t));
    CHECK(t.whole);
    CHECK_EQ(t.first, want);
    CHECK_EQ(t.restart, want == 0x1234);  // the session's first chunk, then each follows on
  }
  CHECK(!r.take(t));
  CHECK(r.push(0x1238));
  CHECK(r.take(t));
  CHECK(t.whole);  // no half sample left over from the drop
  CHECK_EQ(t.first, int16_t(0x1238));
  CHECK(t.restart);  // 0x1237 is missing in between
  CHECK(r.push(0x1239));
  CHECK(r.take(t));
  CHECK(!t.restart);
}

TEST("wake feed: the ring keeps working across the wrap") {
  Rig r(3);
  r.feed.set_session(1);
  Rig::Taken t;
  for (int16_t i = 0; i < 10; ++i) {
    CHECK(r.push(i));
    CHECK(r.take(t));
    CHECK_EQ(t.first, i);
    CHECK_EQ(t.restart, i == 0);
  }
}

TEST("wake feed: re-arming skips audio queued in the earlier session and starts the new one over") {
  Rig r(8);
  r.feed.set_session(1);
  CHECK(r.push(1));
  Rig::Taken t;
  CHECK(r.take(t));
  CHECK(r.push(2));  // still queued when the core disarms and arms again
  CHECK(r.push(3));
  r.feed.set_session(0);
  CHECK(!r.push(4));  // disarmed: not admitted
  r.feed.set_session(3);
  Rig::Taken none;
  CHECK(!r.take(none));  // only the earlier session's audio waits: none of it is handed over
  CHECK(r.push(5));
  CHECK(r.take(t));
  CHECK_EQ(t.first, int16_t(5));
  CHECK_EQ(t.session, uint32_t(3));
  CHECK(t.restart);
  CHECK(!r.take(t));
}

TEST("wake feed: a chunk the producer tagged just before a re-arm is never handed to the new session") {
  Rig r(8);
  r.feed.set_session(1);
  CHECK(r.push(1));  // tagged with session 1; the consumer has not taken it yet
  r.feed.set_session(3);
  CHECK(r.push(2));
  Rig::Taken t;
  CHECK(r.take(t));
  CHECK_EQ(t.first, int16_t(2));
  CHECK_EQ(t.session, uint32_t(3));
}

TEST("wake feed: a detection is current only while its session is the armed one") {
  Rig r(4);
  CHECK(!r.feed.current(0));
  r.feed.set_session(5);
  CHECK(r.feed.current(5));
  CHECK(!r.feed.current(3));
  r.feed.set_session(0);
  CHECK(!r.feed.current(5));
  CHECK(!r.feed.current(0));
}

TEST("wake feed: restart() makes the next chunk start the audio over even though it follows on") {
  Rig r(4);
  r.feed.set_session(1);
  Rig::Taken t;
  CHECK(r.push(1));
  CHECK(r.take(t));
  r.feed.restart();  // a wake-test used the model
  CHECK(r.push(2));
  CHECK(r.take(t));
  CHECK(t.restart);
}
