#!/usr/bin/env python3
"""Verify that a processed Tarnished v6.0 network matches the ARM NEON layout."""
from __future__ import annotations

import sys
from pathlib import Path

INPUT_BUCKETS = 16
OUTPUT_BUCKETS = 8
L1_SIZE = 1792
L2_SIZE = 16
L3_SIZE = 32


def require_equal(label: str, source: bytes, processed: bytes) -> None:
    if source != processed:
        raise SystemExit(f"ERROR: NEON preprocessing mismatch in {label}")
    print(f"PASS: {label}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <quantised-network.bin> <processed.bin>")
    raw = Path(sys.argv[1]).read_bytes()
    net = Path(sys.argv[2]).read_bytes()

    raw_ft_bytes = INPUT_BUCKETS * L1_SIZE * 768 * 2
    ft_bias_bytes = L1_SIZE * 2
    l1_weights_bytes = L1_SIZE * OUTPUT_BUCKETS * L2_SIZE
    l1_bias_bytes = OUTPUT_BUCKETS * L2_SIZE * 4
    l2_weights_bytes = (L2_SIZE * 2) * OUTPUT_BUCKETS * L3_SIZE * 4
    l2_bias_bytes = OUTPUT_BUCKETS * L3_SIZE * 4
    l3_weights_bytes = L3_SIZE * OUTPUT_BUCKETS * 4
    l3_bias_bytes = OUTPUT_BUCKETS * 4

    expected_raw = (raw_ft_bytes + ft_bias_bytes + l1_weights_bytes + l1_bias_bytes
                    + l2_weights_bytes + l2_bias_bytes + l3_weights_bytes + l3_bias_bytes)
    expected_processed = expected_raw + 32  # tail padding from alignas(64) Network
    if len(raw) != expected_raw:
        raise SystemExit(f"ERROR: raw network size {len(raw)} != expected {expected_raw}")
    if len(net) != expected_processed:
        raise SystemExit(f"ERROR: processed network size {len(net)} != expected {expected_processed}")

    raw_off = 0
    net_off = 0
    require_equal("FTWeights (unpermuted ARM order)",
                  raw[raw_off:raw_off + raw_ft_bytes], net[net_off:net_off + raw_ft_bytes])
    raw_off += raw_ft_bytes
    net_off += raw_ft_bytes

    require_equal("FTBiases (unpermuted ARM order)",
                  raw[raw_off:raw_off + ft_bias_bytes], net[net_off:net_off + ft_bias_bytes])
    raw_off += ft_bias_bytes
    net_off += ft_bias_bytes

    src_l1 = raw_off
    dst_l1 = net_off
    # Runtime `forwardL1()` consumes four input activations and 16 signed
    # weights per vector, producing four output neurons in four-byte groups.
    for bucket in range(OUTPUT_BUCKETS):
        for group in range(L1_SIZE // 4):
            for output_neuron in range(L2_SIZE):
                for lane in range(4):
                    src_index = src_l1 + ((group * 4 + lane) * OUTPUT_BUCKETS + bucket) * L2_SIZE + output_neuron
                    dst_index = dst_l1 + bucket * L1_SIZE * L2_SIZE + group * 4 * L2_SIZE + output_neuron * 4 + lane
                    if raw[src_index] != net[dst_index]:
                        raise SystemExit(
                            "ERROR: L1Weights do not match NEON 4-input/4-output dot-product layout "
                            f"(bucket={bucket}, group={group}, output={output_neuron}, lane={lane})"
                        )
    print("PASS: L1Weights (NEON four-byte dot-product layout)")
    raw_off += l1_weights_bytes
    net_off += l1_weights_bytes

    require_equal("L1Biases", raw[raw_off:raw_off + l1_bias_bytes], net[net_off:net_off + l1_bias_bytes])
    raw_off += l1_bias_bytes
    net_off += l1_bias_bytes

    # L2 raw layout [input][bucket][neuron] -> runtime [bucket][input][neuron].
    for bucket in range(OUTPUT_BUCKETS):
        for inp in range(L2_SIZE * 2):
            src = raw_off + (inp * OUTPUT_BUCKETS + bucket) * L3_SIZE * 4
            dst = net_off + (bucket * (L2_SIZE * 2) + inp) * L3_SIZE * 4
            if raw[src:src + L3_SIZE * 4] != net[dst:dst + L3_SIZE * 4]:
                raise SystemExit(
                    f"ERROR: L2Weights mismatch (bucket={bucket}, input={inp})"
                )
    print("PASS: L2Weights (bucket/input transpose)")
    raw_off += l2_weights_bytes
    net_off += l2_weights_bytes

    require_equal("L2Biases", raw[raw_off:raw_off + l2_bias_bytes], net[net_off:net_off + l2_bias_bytes])
    raw_off += l2_bias_bytes
    net_off += l2_bias_bytes

    for bucket in range(OUTPUT_BUCKETS):
        for neuron in range(L3_SIZE):
            src = raw_off + (neuron * OUTPUT_BUCKETS + bucket) * 4
            dst = net_off + (bucket * L3_SIZE + neuron) * 4
            if raw[src:src + 4] != net[dst:dst + 4]:
                raise SystemExit(f"ERROR: L3Weights mismatch (bucket={bucket}, neuron={neuron})")
    print("PASS: L3Weights")
    raw_off += l3_weights_bytes
    net_off += l3_weights_bytes

    require_equal("L3Biases", raw[raw_off:raw_off + l3_bias_bytes], net[net_off:net_off + l3_bias_bytes])
    print("All ARM NEON preprocessing layout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
