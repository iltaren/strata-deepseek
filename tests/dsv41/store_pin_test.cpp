// Strata. ExpertStore pins (nan-fix): a slot found with find_pin() is never given to another expert - not by a read's
// eviction (preload / prefetch / a CPU-lane miss) nor by a swap's put_begin - until unpin(); its bytes stay the expert's;
// gen() moves whenever a slot changes hands; the old unpinned find() is what loses its slot.  No device, a synthetic file.
//   store_pin_test <scratch dir>
#include "ds4_store.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::ds4;

static int fails = 0;
#define CHECK(c, ...)                                         \
    do {                                                      \
        if (!(c)) {                                           \
            ++fails;                                          \
            std::printf("FAIL line %d: %s  ", __LINE__, #c); \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

static constexpr int kExp = 32;
static constexpr uint64_t kMat = 65536;   // each matrix 64 KB, 4096-aligned offsets

static uint32_t word(int r, int e, uint64_t i) { return ((uint32_t) r << 28) ^ ((uint32_t) e << 20) ^ (uint32_t) i ^ 0x9e3779b9u; }

// the slot holds expert e's three matrices exactly
static bool holds(const ExpertStore& st, int s, int e) {
    for (int r = 0; r < 3; ++r) {
        const uint32_t* p = (const uint32_t*) st.mat(s, r);
        for (uint64_t i = 0; i < kMat / 4; ++i)
            if (p[i] != word(r, e, i)) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const std::string path = dir + "/store_pin_test.bin";
    {   // gate of all experts, then up, then down; expert e of matrix r at r * kExp * kMat + e * kMat
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            std::printf("cannot write %s\n", path.c_str());
            return 2;
        }
        std::vector<uint32_t> buf(kMat / 4);
        for (int r = 0; r < 3; ++r)
            for (int e = 0; e < kExp; ++e) {
                for (uint64_t i = 0; i < kMat / 4; ++i) buf[i] = word(r, e, i);
                std::fwrite(buf.data(), 4, buf.size(), f);
            }
        std::fclose(f);
    }
    const os::fd_t dfd = os::open_direct(path), pfd = os::open_par(path);
    ExpertStore::LayerGeo geo;
    geo.n_exp = kExp;
    uint64_t at = 0;
    for (int r = 0; r < 3; ++r) {
        geo.file[r] = 0;
        geo.off[r] = (uint64_t) r * kExp * kMat;
        geo.stride[r] = kMat;
        geo.bytes[r] = kMat;
        geo.tr[r] = at;
        at += kMat + ExpertStore::kSlack;
    }
    geo.blob = at;
    ExpertStore st;
    ExpertStore::Cfg cfg;
    cfg.threads = 2;
    cfg.floor_bytes = 0;
    const size_t slot = (geo.blob + 2 * ExpertStore::kSlack + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
    cfg.budget_bytes = 4 * slot;   // four slots: every new read evicts
    if (!st.init(cfg, {geo}, {dfd}, {pfd})) {
        std::printf("init failed\n");
        return 2;
    }
    CHECK(st.slots_max() == 4, "slots %zu", st.slots_max());
    auto fill = [&](std::vector<int> es) {   // reads them (each evicting the coldest unpinned slot), waits
        std::vector<std::pair<int, int>> l;
        for (int e : es) l.push_back({0, e});
        st.preload(l, 0);
    };
    // 1. pinned: survives reads' evictions and swaps' put_begin, bytes intact
    fill({0, 1, 2, 3});
    const int s0 = st.find_pin(0, 0);
    CHECK(s0 >= 0, "expert 0 not found after preload");
    const uint32_t g0 = s0 >= 0 ? st.gen(s0) : 0;
    for (int round = 0; round < 8; ++round) fill({4 + round * 3, 5 + round * 3, 6 + round * 3});
    for (int e = 28; e < 32; ++e) {   // swaps handing experts back: put_begin takes a slot, the caller fills it
        uint8_t* dst = st.put_begin(0, e);
        if (dst == nullptr) continue;
        for (int r = 0; r < 3; ++r) {
            uint32_t* p = (uint32_t*) (dst + geo.tr[r]);
            for (uint64_t i = 0; i < kMat / 4; ++i) p[i] = word(r, e, i);
        }
        st.put_done(0, e, true);
    }
    CHECK(s0 >= 0 && st.gen(s0) == g0, "pinned slot changed hands: gen %u -> %u", g0, s0 >= 0 ? st.gen(s0) : 0);
    CHECK(s0 >= 0 && st.find(0, 0) == s0, "pinned expert 0 no longer findable in its slot");
    CHECK(s0 >= 0 && holds(st, s0, 0), "pinned slot's bytes are not expert 0's");
    // 2. nested pins: one unpin of two keeps it
    CHECK(st.find_pin(0, 0) == s0, "second pin found another slot");
    st.unpin(s0);
    for (int round = 0; round < 6; ++round) fill({10 + round, 20 + round});
    CHECK(st.gen(s0) == g0 && holds(st, s0, 0), "slot lost with one pin still held");
    // 3. unpinned: evictable again (the slot changes hands, gen says so)
    st.unpin(s0);
    bool moved = false;
    for (int round = 0; round < 20 && !moved; ++round) {
        fill({(round * 5) % 28 + 1, (round * 5 + 2) % 28 + 1, (round * 5 + 4) % 28 + 1});
        moved = st.gen(s0) != g0;
    }
    CHECK(moved, "unpinned slot never evicted");
    // 4. the old way (find, no pin): the pointer outlives the slot's owner - what the prompt path's stream did
    fill({0});
    const int s1 = st.find(0, 0);
    CHECK(s1 >= 0, "expert 0 not found");
    const uint32_t g1 = s1 >= 0 ? st.gen(s1) : 0;
    bool stale = false;
    for (int round = 0; round < 20 && !stale && s1 >= 0; ++round) {
        fill({(round * 7) % 27 + 2, (round * 7 + 3) % 27 + 2, (round * 7 + 5) % 27 + 2});
        stale = st.gen(s1) != g1;
    }
    CHECK(stale, "unpinned find's slot was never reassigned (the race not shown)");
    if (stale && s1 >= 0)
        std::printf("old unpinned pattern: slot %d now holds layer %d expert %d, its bytes expert 0's: %s\n", s1, st.holder_layer(s1),
                    st.holder_expert(s1), holds(st, s1, 0) ? "yes" : "NO - another expert's");
    std::remove(path.c_str());
    std::printf(fails == 0 ? "store_pin_test: all checks passed\n" : "store_pin_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
