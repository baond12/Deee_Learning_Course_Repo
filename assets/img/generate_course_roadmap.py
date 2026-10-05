from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
OUT = Path(__file__).with_name("course-roadmap.png")


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


pixels = bytearray()
for y in range(HEIGHT):
    row = bytearray([0])
    for x in range(WIDTH):
        tx = x / (WIDTH - 1)
        ty = y / (HEIGHT - 1)
        base = blend((11, 31, 56), (10, 116, 107), tx * 0.55 + ty * 0.2)
        light = int(24 * math.sin((x + y) / 95) + 18 * math.sin(x / 57))
        row.extend(max(0, min(255, c + light)) for c in base)
    pixels.extend(row)


def set_px(x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        idx = y * (WIDTH * 3 + 1) + 1 + x * 3
        pixels[idx : idx + 3] = bytes(color)


def circle(cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    rr = radius * radius
    for yy in range(cy - radius, cy + radius + 1):
        for xx in range(cx - radius, cx + radius + 1):
            if (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= rr:
                set_px(xx, yy, color)


def line(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    x, y = x1, y1
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                set_px(x + ox, y + oy, color)
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


nodes = [
    (980, 180),
    (1200, 250),
    (1370, 160),
    (1090, 430),
    (1320, 460),
    (1430, 650),
    (1130, 690),
    (920, 570),
]

for i, a in enumerate(nodes):
    for b in nodes[i + 1 :]:
        if abs(a[0] - b[0]) < 360:
            line(a[0], a[1], b[0], b[1], (125, 211, 203))

for idx, (x, y) in enumerate(nodes):
    circle(x, y, 28 if idx % 2 else 34, (245, 248, 255))
    circle(x, y, 17 if idx % 2 else 20, (180, 35, 24) if idx in (2, 5) else (15, 118, 110))

for x, y, w, h, color in [
    (805, 675, 34, 90, (245, 248, 255)),
    (855, 618, 34, 147, (158, 202, 199)),
    (905, 565, 34, 200, (245, 248, 255)),
    (955, 638, 34, 127, (180, 35, 24)),
]:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            set_px(xx, yy, color)

raw = bytes(pixels)
png = bytearray(b"\x89PNG\r\n\x1a\n")


def chunk(kind: bytes, data: bytes) -> None:
    png.extend(struct.pack(">I", len(data)))
    png.extend(kind)
    png.extend(data)
    png.extend(struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
chunk(b"IDAT", zlib.compress(raw, 9))
chunk(b"IEND", b"")
OUT.write_bytes(png)
print(OUT)

