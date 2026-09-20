#pragma once

#include <bitset>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <sys/mman.h>

#include "move.h"
#include "board.h"

constexpr size_t TT_ALIGNMENT = 64;

inline void* alignedAlloc(size_t alignment, size_t requiredBytes) {
    void* ptr = nullptr;

#if defined(__MINGW32__)

    const size_t offset = alignment - 1;
    void* p = std::malloc(requiredBytes + offset);

    if (p != nullptr) {
        ptr = reinterpret_cast<void*>(
            (reinterpret_cast<size_t>(p) + offset) & ~(alignment - 1)
        );
    }

#elif defined(__ANDROID__) || defined(__linux__)

    // Android/bionic and libc++ do not reliably expose
    // std::aligned_alloc. POSIX aligned allocation works on
    // Android and guarantees the requested alignment.
    if (posix_memalign(&ptr, alignment, requiredBytes) != 0)
        ptr = nullptr;

#elif defined(__GNUC__)

    ptr = std::aligned_alloc(alignment, requiredBytes);

#else

#error "Compiler not supported"

#endif

#if defined(__linux__)
    if (ptr != nullptr)
        madvise(ptr, requiredBytes, MADV_HUGEPAGE);
#endif

    return ptr;
}

class TranspositionTable {
   private:
    struct TTCluster {
        static constexpr int ClusterSize = 3;

        struct TTEntry {
            uint64_t data;
        };

        TTEntry entry[ClusterSize];
    };

    TTCluster* table = nullptr;
    size_t clusterCount = 0;

    std::atomic<bool> stopClear{false};
    std::thread clearThread;

   public:
    TranspositionTable() = default;

    ~TranspositionTable() {
        stopClear.store(true, std::memory_order_relaxed);

        if (clearThread.joinable())
            clearThread.join();

        if (table != nullptr)
            std::free(table);
    }

    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    void resize(size_t mb) {
        if (table != nullptr) {
            std::free(table);
            table = nullptr;
        }

        const size_t bytes = mb * 1024ULL * 1024ULL;
        clusterCount = bytes / sizeof(TTCluster);

        if (clusterCount == 0)
            return;

        // Keep the number of clusters a power of two so that
        // TT indexing can use a fast bit mask.
        size_t power = 1;

        while ((power << 1) <= clusterCount)
            power <<= 1;

        clusterCount = power;

        const size_t requiredBytes =
            clusterCount * sizeof(TTCluster);

        table = static_cast<TTCluster*>(
            alignedAlloc(TT_ALIGNMENT, requiredBytes)
        );

        if (table == nullptr) {
            clusterCount = 0;
            throw std::bad_alloc();
        }

        std::memset(table, 0, requiredBytes);
    }

    void clear() {
        if (table == nullptr || clusterCount == 0)
            return;

        const size_t bytes =
            clusterCount * sizeof(TTCluster);

        std::memset(table, 0, bytes);
    }

    void clearAsync() {
        if (clearThread.joinable())
            clearThread.join();

        stopClear.store(false, std::memory_order_relaxed);

        clearThread = std::thread([this]() {
            if (table == nullptr || clusterCount == 0)
                return;

            const size_t bytes =
                clusterCount * sizeof(TTCluster);

            constexpr size_t chunkSize = 1ULL << 20;

            char* ptr = reinterpret_cast<char*>(table);

            size_t offset = 0;

            while (offset < bytes) {
                if (stopClear.load(std::memory_order_relaxed))
                    return;

                const size_t remaining = bytes - offset;
                const size_t amount =
                    std::min(chunkSize, remaining);

                std::memset(ptr + offset, 0, amount);

                offset += amount;
            }
        });
    }

    TTCluster* getCluster(uint64_t key) {
        if (table == nullptr || clusterCount == 0)
            return nullptr;

        return &table[key & (clusterCount - 1)];
    }

    const TTCluster* getCluster(uint64_t key) const {
        if (table == nullptr || clusterCount == 0)
            return nullptr;

        return &table[key & (clusterCount - 1)];
    }

    size_t hashfull() const {
        if (table == nullptr || clusterCount == 0)
            return 0;

        size_t used = 0;
        size_t total = 0;

        // Sample the table rather than scanning the entire TT.
        constexpr size_t samples = 1000;

        const size_t step =
            std::max<size_t>(1, clusterCount / samples);

        for (size_t i = 0; i < clusterCount && total < samples; i += step) {
            const TTCluster& cluster = table[i];

            for (int j = 0; j < TTCluster::ClusterSize; ++j) {
                if (cluster.entry[j].data != 0)
                    ++used;

                ++total;
            }
        }

        if (total == 0)
            return 0;

        return (used * 1000) / total;
    }

    size_t size() const {
        return clusterCount;
    }

    bool empty() const {
        return table == nullptr || clusterCount == 0;
    }
};
