"""
Generate icon assets for all platforms from the GuildMaster medallion logo.

Run once from the repo root before building:
    python scripts/generate_icon.py

Outputs:
    assets/icon.ico        — Windows (.exe embedded icon, all sizes in one file)
    assets/icon.png        — Linux   (referenced by assets/guildmaster.desktop)
    assets/icon.icns       — macOS   (app bundle icon, requires macOS or iconutil)

Requires Pillow:
    pip install Pillow
"""

import math
import os
import struct
import sys
import zlib
from PIL import Image, ImageDraw

# Colours matching constants.hpp
PANEL   = (28,  20,   8, 255)
GOLD    = (200, 145,  40, 255)
GOLD_LT = (240, 192,  80, 255)
BORDER  = (80,   55,  22, 255)


def draw_logo(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2.0
    radius = size / 2.0 * 0.94

    # Background circle + gold ring border
    r = radius
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=PANEL, outline=GOLD,
                 width=max(1, round(r * 0.07)))

    # Subtle inner ring
    r2 = radius * 0.82
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2],
                 fill=None, outline=BORDER,
                 width=max(1, round(radius * 0.025)))

    # 4-pointed star (8 alternating outer/inner vertices)
    outer_r = radius * 0.52
    inner_r = radius * 0.20
    pts = []
    for i in range(8):
        angle = math.radians(i * 45.0 - 90.0)
        r_i   = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r_i * math.cos(angle),
                    cy + r_i * math.sin(angle)))
    draw.polygon(pts, fill=GOLD_LT)

    # Centre gem
    gr = radius * 0.12
    draw.ellipse([cx - gr, cy - gr, cx + gr, cy + gr],
                 fill=PANEL, outline=GOLD,
                 width=max(1, round(radius * 0.03)))

    return img


SIZES = [16, 32, 48, 64, 128, 256]

os.makedirs("assets", exist_ok=True)
images = {s: draw_logo(s) for s in SIZES}

# ── Windows: .ico ─────────────────────────────────────────────────────────────
img_list = [images[s] for s in SIZES]
img_list[0].save(
    "assets/icon.ico",
    format="ICO",
    sizes=[(s, s) for s in SIZES],
    append_images=img_list[1:],
)
print("Written assets/icon.ico")

# ── Linux: 256px PNG ──────────────────────────────────────────────────────────
images[256].save("assets/icon.png", format="PNG")
print("Written assets/icon.png")

# ── macOS: .icns ──────────────────────────────────────────────────────────────
# .icns is a simple container: magic + entries, each entry = OSType tag + data.
# We write ic08 (128px), ic09 (512px — upscaled), ic07 (128px small).
# For a proper build pipeline use iconutil on macOS; this is a portable fallback.
ICNS_TAGS = [
    ("icp4",  16),
    ("icp5",  32),
    ("icp6",  64),
    ("ic07", 128),
    ("ic08", 256),
]

def png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

entries = []
for tag, size in ICNS_TAGS:
    data = png_bytes(images[size])
    # Each entry: 4-byte OSType + 4-byte length (including the 8-byte header)
    entries.append(tag.encode("ascii") + struct.pack(">I", len(data) + 8) + data)

body = b"".join(entries)
total_len = 8 + len(body)  # 8 bytes for the file header
with open("../assets/icon.icns", "wb") as f:
    f.write(b"icns")                      # magic
    f.write(struct.pack(">I", total_len)) # total file length
    f.write(body)
print("Written assets/icon.icns")

# ── Linux .desktop file ───────────────────────────────────────────────────────
# Install to ~/.local/share/applications/ and update-desktop-database to register.
desktop = """\
[Desktop Entry]
Type=Application
Name=GuildMaster
Comment=AI-powered fantasy guild manager
Exec={exec}
Icon={icon}
Categories=Game;
Terminal=false
"""
# Use absolute paths so the .desktop works from any working directory.
exec_path  = os.path.abspath("image_generator")
icon_path  = os.path.abspath("assets/icon.png")
with open("../assets/guildmaster.desktop", "w") as f:
    f.write(desktop.format(exec=exec_path, icon=icon_path))
print("Written assets/guildmaster.desktop")
print()
print("Linux install:")
print("  cp assets/guildmaster.desktop ~/.local/share/applications/")
print("  update-desktop-database ~/.local/share/applications/")
