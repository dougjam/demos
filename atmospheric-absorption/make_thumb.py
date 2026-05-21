"""
Generate atmospheric-absorption/thumb.png — a small log-log plot of
alpha(f) at default conditions, in the style of the other demo gallery
thumbnails (dark background, accent line).

Run:
    python make_thumb.py
"""
from pathlib import Path
import math
import struct
import zlib

from python.iso9613_reference import alpha_dB_per_km

# Try matplotlib first (cleaner output); fall back to a pure-Python PNG.
W, H = 400, 240


def make_with_matplotlib() -> bytes | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fs = [10 ** (1 + 4 * i / 255) for i in range(256)]
    a = [alpha_dB_per_km(f, 20, 50, 101.325) for f in fs]
    fig = plt.figure(figsize=(4, 2.4), dpi=100, facecolor="#0a0a1e")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0a0a1e")
    ax.loglog(fs, a, color="#2563eb", linewidth=3)
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-2, 1e4)
    for spine in ax.spines.values():
        spine.set_color("#445")
    ax.tick_params(colors="#778", which="both", labelsize=8)
    ax.grid(True, which="both", color="#223", linewidth=0.5)
    ax.set_xlabel("freq", color="#aab", fontsize=8)
    ax.set_ylabel(r"$\alpha$ (dB/km)", color="#aab", fontsize=8)
    fig.tight_layout(pad=0.4)
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="#0a0a1e")
    plt.close(fig)
    return buf.getvalue()


def make_fallback() -> bytes:
    # Pure-Python PNG: dark background with a single blue curve.
    fs = [10 ** (1 + 4 * i / (W - 1)) for i in range(W)]
    a = [alpha_dB_per_km(f, 20, 50, 101.325) for f in fs]
    log_a_min, log_a_max = -2, 4
    def y_pix(v):
        lv = math.log10(max(1e-3, v))
        t = (lv - log_a_min) / (log_a_max - log_a_min)
        return int((1 - t) * (H - 1))

    bg = (10, 10, 30)
    line = (37, 99, 235)
    pixels = [list(bg) * W for _ in range(H)]
    for x in range(W):
        y = y_pix(a[x])
        for dy in (-1, 0, 1):
            yy = max(0, min(H - 1, y + dy))
            row = pixels[yy]
            row[3 * x] = line[0]; row[3 * x + 1] = line[1]; row[3 * x + 2] = line[2]
    # Encode as PNG.
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        raw.extend(row)
    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data)
        return out + struct.pack(">I", crc)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


if __name__ == "__main__":
    out = Path(__file__).parent / "thumb.png"
    data = make_with_matplotlib() or make_fallback()
    out.write_bytes(data)
    print(f"wrote {out}  ({len(data)} bytes)")
