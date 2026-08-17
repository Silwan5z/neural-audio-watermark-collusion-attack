"""Shortened Hamming codes used for the single-bit-correction attack study.

For a d-bit watermark, p=ceil(log2(d+1)) parity locations (1,2,4,...)
give a shortened Hamming (d, d-p) code.  Its minimum distance is three,
so a decoder corrects every single corrupted watermark bit.
"""
from __future__ import annotations

import numpy as np


def parity_bits(code_bits: int) -> int:
    """Smallest p such that a d-bit shortened Hamming code corrects 1 bit."""
    if code_bits < 3:
        raise ValueError("at least three code bits are required")
    p = 0
    while (1 << p) < code_bits + 1:
        p += 1
    return p


def info_bits(code_bits: int) -> int:
    return code_bits - parity_bits(code_bits)


def encode_bits(message_bits, code_bits: int) -> np.ndarray:
    """Encode LSB-first message bits into a LSB-first shortened Hamming word."""
    k = info_bits(code_bits)
    message_bits = np.asarray(message_bits, dtype=np.int8)
    if message_bits.shape != (k,):
        raise ValueError(f"expected {k} message bits, got {message_bits.shape}")
    word = np.zeros(code_bits, dtype=np.int8)
    data_positions = [pos for pos in range(1, code_bits + 1) if pos & (pos - 1)]
    word[np.array(data_positions) - 1] = message_bits
    for parity_pos in (1 << j for j in range(parity_bits(code_bits))):
        word[parity_pos - 1] = sum(word[pos - 1] for pos in range(1, code_bits + 1)
                                   if pos & parity_pos) & 1
    return word


def decode_bits(received_bits, code_bits: int) -> tuple[np.ndarray, int]:
    """Correct one bit if its Hamming syndrome names a retained position.

    Returns `(message_bits, syndrome)`.  A syndrome beyond `code_bits` signals
    an uncorrectable pattern in the shortened code and is left unchanged.
    """
    word = np.asarray(received_bits, dtype=np.int8).copy()
    if word.shape != (code_bits,):
        raise ValueError(f"expected {code_bits} code bits, got {word.shape}")
    syndrome = 0
    for parity_pos in (1 << j for j in range(parity_bits(code_bits))):
        if sum(word[pos - 1] for pos in range(1, code_bits + 1) if pos & parity_pos) & 1:
            syndrome |= parity_pos
    if 1 <= syndrome <= code_bits:
        word[syndrome - 1] ^= 1
    data_positions = [pos for pos in range(1, code_bits + 1) if pos & (pos - 1)]
    return word[np.array(data_positions) - 1], syndrome


def int_to_bits(value: int, nbits: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(nbits)], dtype=np.int8)


def bits_to_int(bits) -> int:
    bits = np.asarray(bits, dtype=np.int8)
    return int(bits @ (1 << np.arange(len(bits))))


def encode_int(message: int, code_bits: int) -> int:
    return bits_to_int(encode_bits(int_to_bits(message, info_bits(code_bits)), code_bits))


def ecc_codebook(code_bits: int) -> np.ndarray:
    """Rows are codewords; row index is the underlying information message."""
    n = 1 << info_bits(code_bits)
    return np.stack([encode_bits(int_to_bits(msg, info_bits(code_bits)), code_bits)
                     for msg in range(n)])
