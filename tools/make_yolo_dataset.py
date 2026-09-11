"""Turn the labelled pool into a YOLO dataset with a train/val split.

The labeller writes one flat folder of images and labels because that is the
easiest thing to resume; ultralytics wants images/<split> and labels/<split>.
This also doubles as a quality gate: every box is checked for sane normalised
coordinates, and the aspect ratio is reported, because the target is a circle
and a label whose box is far from square is a mislabelled frame.

Usage:
    python tools/make_yolo_dataset.py            # writes %USERPROFILE%\\fisheye_yolo
"""

import os
import shutil
import sys

POOL = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_dataset")
OUT = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_yolo")
EVERY_NTH_TO_VAL = 5


def read_label(path):
    with open(path) as handle:
        parts = handle.read().split()
    if len(parts) != 5:
        return None
    try:
        return [float(v) for v in parts]
    except ValueError:
        return None


def main():
    images_dir = os.path.join(POOL, "images")
    labels_dir = os.path.join(POOL, "labels")
    if not os.path.isdir(images_dir):
        print("no labelled pool at", POOL)
        sys.exit(1)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    problems = []
    for stem in stems:
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            problems.append(f"{stem}: 没有标签")
            continue
        values = read_label(label_path)
        if values is None:
            problems.append(f"{stem}: 标签格式不对")
            continue
        cls, cx, cy, bw, bh = values
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            problems.append(f"{stem}: 坐标越界 {values}")

    usable = [s for s in stems if not any(p.startswith(s + ":") for p in problems)]
    if not usable:
        print("no usable frames")
        for p in problems:
            print("  ", p)
        sys.exit(1)

    # Deterministic split: no shuffle dependency, so a rerun reproduces it.
    val = {s for i, s in enumerate(usable) if i % EVERY_NTH_TO_VAL == 0}
    train = [s for s in usable if s not in val]

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split))
        os.makedirs(os.path.join(OUT, "labels", split))

    def copy(stem, split):
        image = next(os.path.join(images_dir, stem + ext)
                     for ext in (".jpg", ".jpeg", ".png")
                     if os.path.exists(os.path.join(images_dir, stem + ext)))
        shutil.copyfile(image, os.path.join(OUT, "images", split, os.path.basename(image)))
        shutil.copyfile(os.path.join(labels_dir, stem + ".txt"),
                        os.path.join(OUT, "labels", split, stem + ".txt"))

    for stem in train:
        copy(stem, "train")
    for stem in sorted(val):
        copy(stem, "val")

    ratios = []
    for stem in usable:
        _, _, _, bw, bh = read_label(os.path.join(labels_dir, stem + ".txt"))
        ratios.append(bw / bh if bh else 0.0)
    ratios.sort()
    median = ratios[len(ratios) // 2]

    with open(os.path.join(OUT, "data.yaml"), "w") as handle:
        handle.write("path: %s\n" % OUT.replace("\\", "/"))
        handle.write("train: images/train\n")
        handle.write("val: images/val\n")
        handle.write("nc: 1\n")
        handle.write("names:\n  0: screen\n")

    print(f"数据集: {OUT}")
    print(f"训练 {len(train)} 张 / 验证 {len(val)} 张")
    print(f"框宽高比 中位数 {median:.2f}  最小 {ratios[0]:.2f}  最大 {ratios[-1]:.2f}")
    if problems:
        print("有问题:")
        for p in problems:
            print("  ", p)
    else:
        print("标签检查: 全部通过")
    print("偏方的框说明标歪了，宽高比离 1.0 太远的那几张值得回头看一眼")


if __name__ == "__main__":
    main()
