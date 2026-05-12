#!/usr/bin/env python3
import sys
from PIL import Image

fn = sys.argv[1]
max_w = int(sys.argv[2])
max_h = int(sys.argv[3])

img = Image.open(fn)

while img.width >= max_w or img.height >= max_h:
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
if img.mode in ("RGBA", "LA", "P"):
    img = img.convert("RGBA").convert("RGB")
img.save(fn)
