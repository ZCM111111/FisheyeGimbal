"""Headless smoke test for the labeller: proposals, boxes, resume logic.

No window is opened, so this can run anywhere.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cabinet_detect  # noqa: E402
import label_tool  # noqa: E402

SOURCES = [r"D:\桌面文件\训练2", r"D:\桌面文件\训练数据"]
OUT = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_dataset")

files = []
for source in SOURCES:
    if not os.path.isdir(source):
        print("missing:", source)
        continue
    names = sorted(n for n in os.listdir(source)
                   if n.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"{len(names):3d}  {source}")
    files += [os.path.join(source, n) for n in names]

print(f"total {len(files)} frames")

labels_dir = os.path.join(OUT, "labels")
already = [i for i, p in enumerate(files)
           if os.path.exists(os.path.join(labels_dir, label_tool.stem_for(i, p) + ".txt"))]
print(f"already labelled: {len(already)} -> resume at index "
      f"{min([i for i in range(len(files)) if i not in already], default=0)}")

# Exercise the geometry helpers the keys drive.
w, h = 1080, 1920
box = label_tool.box_from_circle(540, 960, 300, w, h)
print("box        ", tuple(round(v) for v in box))
print("move right ", tuple(round(v) for v in label_tool.move_box(box, "right", 5, w, h)))
print("grow       ", tuple(round(v) for v in label_tool.resize_box(box, 5, w, h)))
print("normalised ", tuple(round(v, 4) for v in label_tool.normalise(box, w, h)))

# Proposals on a few real frames: the top candidate should be a sane circle.
for path in files[::max(1, len(files) // 6)][:6]:
    image = cabinet_detect.imread_unicode(path)
    if image is None:
        print("unreadable:", os.path.basename(path))
        continue
    found = label_tool.proposals(image)
    ih, iw = image.shape[:2]
    if not found:
        print(f"{os.path.basename(path):32s} no candidates")
        continue
    top = found[0]
    inside = (0 <= top["cx"] - top["r"] and top["cx"] + top["r"] <= iw
              and 0 <= top["cy"] - top["r"] and top["cy"] + top["r"] <= ih)
    print(f"{os.path.basename(path):32s} {len(found)} cand  "
          f"r={top['r']:.0f} ({top['r'] / min(iw, ih):.2f} of short side)  "
          f"support={top['support']:.2f}  fully inside={inside}")
