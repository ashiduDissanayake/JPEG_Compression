#!/usr/bin/env python3
import sys
import os
import heapq
import numpy as np
from PIL import Image

ZIGZAG_INDEX = np.array([
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
], dtype=np.int32)

QY_50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float32)

QC_50 = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float32)

def load_image_as_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def rgb_to_ycbcr(rgb):
    rgb = rgb.astype(np.float32)
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)
    ycbcr = np.dot(rgb, M.T)
    ycbcr[:, :, 1] += 128.0
    ycbcr[:, :, 2] += 128.0
    return np.clip(ycbcr, 0, 255).astype(np.uint8)

def chroma_subsample_420(ycbcr):
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]
    h, w = Cb.shape
    Cb = Cb[:(h//2)*2, :(w//2)*2]
    Cr = Cr[:(h//2)*2, :(w//2)*2]
    Cb_sub = Cb.reshape(Cb.shape[0]//2, 2, Cb.shape[1]//2, 2).mean(axis=(1,3)).astype(Cb.dtype)
    Cr_sub = Cr.reshape(Cr.shape[0]//2, 2, Cr.shape[1]//2, 2).mean(axis=(1,3)).astype(Cr.dtype)
    return Y, Cb_sub, Cr_sub

def split_into_blocks(channel, block_size=8):
    h, w = channel.shape
    nh = h // block_size
    nw = w // block_size
    blocks = channel.reshape(nh, block_size, nw, block_size).swapaxes(1,2)
    return blocks

def dct2d_separable(block):
    N = 8
    T = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        alpha = np.sqrt(1.0/N) if i == 0 else np.sqrt(2.0/N)
        for j in range(N):
            T[i,j] = alpha * np.cos((2*j+1) * i * np.pi / (2*N))
    return T @ block @ T.T

def apply_dct_to_blocks(blocks):
    nh, nw, _, _ = blocks.shape
    out = np.zeros((nh, nw, 8, 8), dtype=np.float32)
    for i in range(nh):
        for j in range(nw):
            b = blocks[i,j].astype(np.float32)
            out[i,j] = dct2d_separable(b - 128.0)
    return out

def scale_quantization_table(Q_base, quality):
    q = max(1, min(int(quality), 100))
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2.0 * q
    Q = np.floor((Q_base * scale + 50.0) / 100.0)
    return np.clip(Q, 1, 255).astype(np.float32)

def quantize_blocks(dct_blocks, Q_table):
    nh, nw, _, _ = dct_blocks.shape
    quant = np.zeros_like(dct_blocks, dtype=np.int32)
    for i in range(nh):
        for j in range(nw):
            quant[i,j] = np.round(dct_blocks[i,j] / Q_table).astype(np.int32)
    return quant

def zigzag_encode(block):
    arr = np.zeros(64, dtype=block.dtype)
    for i in range(8):
        for j in range(8):
            arr[ZIGZAG_INDEX[i,j]] = block[i,j]
    return arr

def ac_run_length_encode(zigzag_sequence):
    rle = []
    run = 0
    for k in range(1,64):
        v = int(zigzag_sequence[k])
        if v == 0:
            run += 1
        else:
            while run > 15:
                rle.append((15,0))
                run -= 16
            rle.append((run, v))
            run = 0
    if run > 0 or len(rle) == 0:
        rle.append((0,0))
    return rle

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq
    def is_leaf(self):
        return self.symbol is not None

def build_huffman_tree(freqs):
    if not freqs:
        return None
    if len(freqs) == 1:
        sym = next(iter(freqs.keys()))
        return HuffmanNode(symbol=sym, freq=freqs[sym])
    heap = [HuffmanNode(symbol=s, freq=f) for s,f in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = HuffmanNode(freq=a.freq + b.freq, left=a, right=b)
        heapq.heappush(heap, parent)
    return heap[0]

def generate_huffman_codes(root, code="", codes=None):
    if codes is None:
        codes = {}
    if root is None:
        return codes
    if root.is_leaf():
        codes[root.symbol] = code if code else "0"
        return codes
    generate_huffman_codes(root.left, code + "0", codes)
    generate_huffman_codes(root.right, code + "1", codes)
    return codes

def pipeline_and_save(image_rgb, quality, out_path):
    ycbcr = rgb_to_ycbcr(image_rgb)
    Y, Cb_sub, Cr_sub = chroma_subsample_420(ycbcr)
    Y_blocks = split_into_blocks(Y)
    Cb_blocks = split_into_blocks(Cb_sub)
    Cr_blocks = split_into_blocks(Cr_sub)
    Y_dct = apply_dct_to_blocks(Y_blocks)
    Cb_dct = apply_dct_to_blocks(Cb_blocks)
    Cr_dct = apply_dct_to_blocks(Cr_blocks)
    QY = scale_quantization_table(QY_50, quality)
    QC = scale_quantization_table(QC_50, quality)
    Y_quant = quantize_blocks(Y_dct, QY)
    Cb_quant = quantize_blocks(Cb_dct, QC)
    Cr_quant = quantize_blocks(Cr_dct, QC)

    total = Y_quant.size + Cb_quant.size + Cr_quant.size
    zeros = int(np.sum(Y_quant == 0) + np.sum(Cb_quant == 0) + np.sum(Cr_quant == 0))
    print(f"Quality={quality}: zero coeffs {zeros:,}/{total:,} ({zeros/total*100:.1f}%)")

    # Zigzag + RLE across all blocks -> collect AC symbols
    freqs = {}
    def collect_from_blocks(blocks):
        nh, nw, _, _ = blocks.shape
        for i in range(nh):
            for j in range(nw):
                zz = zigzag_encode(blocks[i,j])
                rle = ac_run_length_encode(zz)
                for sym in rle:
                    freqs[sym] = freqs.get(sym, 0) + 1

    collect_from_blocks(Y_quant)
    collect_from_blocks(Cb_quant)
    collect_from_blocks(Cr_quant)

    # Build Huffman codes from frequencies
    hroot = build_huffman_tree(freqs)
    codes = generate_huffman_codes(hroot)

    print(f"  Distinct AC symbols: {len(freqs)}")
    # show a few Huffman codes (up to 10)
    sample = list(codes.items())[:10]
    for sym, code in sample:
        print(f"    {sym} -> {code}")

    # Save real JPEG using Pillow
    pil = Image.fromarray(image_rgb)
    try:
        pil.save(out_path, format='JPEG', quality=int(quality), optimize=True)
    except OSError:
        pil.save(out_path, format='JPEG', quality=int(quality))
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"  Saved JPEG: {out_path} ({size_kb:.1f} KB)")

    return {
        "Y_quant": Y_quant, "Cb_quant": Cb_quant, "Cr_quant": Cr_quant,
        "QY": QY, "QC": QC, "freqs": freqs, "codes": codes
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_image>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    img = load_image_as_rgb(path)
    h, w = img.shape[:2]
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    if new_h != h or new_w != w:
        img = img[:new_h, :new_w, :]

    qualities = [10, 30, 50, 70, 90]
    out_dir = "compressed_outputs"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    for q in qualities:
        out_file = os.path.join(out_dir, f"{base}_q{q}.jpg")
        pipeline_and_save(img, q, out_file)

if __name__ == "__main__":
    main()
