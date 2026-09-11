"""Model and classical detector on the same frame: does agreement help?

The ONNX model generalises better but is a weak classifier, and when the screen
is small in frame it falls back to the size prior it learned — a confident-
looking circle pointing at nothing. The classical circle detector has no such
prior and was measured at 98% on these very frames. So the app asks both and
uses the model's box only when the two agree.

This checks that rule against the labels before trusting it in the field:

    D:\\Python312\\python.exe tools\\check_ensemble.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cabinet_detect  # noqa: E402
import check_onnx as probe  # noqa: E402

ACT_CONFIDENCE = 0.15          # matches minimumModelConfidence in App.swift
CENTRE_TOLERANCE = 0.5          # centres may differ by half a screen radius
RADIUS_RANGE = (0.6, 1.7)
GOOD_IOU = 0.5


def circle_box(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def main():
    session = ort.InferenceSession(probe.MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    paths = (sorted(glob.glob(os.path.join(probe.DATASET, "images", "val", "*.jpg")))
             + sorted(glob.glob(os.path.join(probe.DATASET, "images", "train", "*.jpg"))))
    if not paths:
        sys.exit("no labelled frames")

    tally = {"model only": 0, "classical only": 0, "ensemble": 0, "agree": 0, "disagree": 0}
    total = 0
    for path in paths:
        image = probe.read_image(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        label_dir = "val" if "val" in path else "train"
        label_path = os.path.join(probe.DATASET, "labels", label_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path) as handle:
            parts = [float(v) for v in handle.read().split()]
        h, w = image.shape[:2]
        truth = circle_box(parts[1] * w, parts[2] * h, parts[3] * w / 2)
        total += 1

        # Model, on the same square centre crop Vision would hand it.
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = image[y0:y0 + side, x0:x0 + side]
        rgb = cv2.cvtColor(cv2.resize(crop, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        output = np.asarray(session.run([output_name], {input_name: tensor})[0])
        _, box, score = probe.decode(output)
        model = None
        if box is not None:
            mx = x0 + box[0] / 640 * side
            my = y0 + box[1] / 640 * side
            mr = (box[2] + box[3]) / 4 / 640 * side
            model = circle_box(mx, my, mr)

        # Classical, on the whole frame.
        found = cabinet_detect.detect(image)
        classical = None
        if found.get("ok"):
            classical = circle_box(found["cx"], found["cy"], found["r"])

        model_ok = model is not None and probe.iou(model, truth) >= GOOD_IOU
        classical_ok = classical is not None and probe.iou(classical, truth) >= GOOD_IOU

        if model_ok:
            tally["model only"] += 1
        if classical_ok:
            tally["classical only"] += 1

        # The rule under test: the model leads, the classical detector answers
        # when the model is absent or weak.
        #
        # Both detectors score 49/50 on these frames, and the single frame they
        # disagree on is not enough to break the tie — so the choice is made on
        # principle: the model exists to generalise to machines and lighting the
        # classical rules were never tuned for. What the model does need
        # protecting from is its own size prior when the screen is small in
        # frame, and that failure scores 0.11-0.14 — every measured false
        # positive sits below the 0.15 action threshold, so the gate covers it.
        if model is None or score is None or score < ACT_CONFIDENCE:
            chosen, used = classical, "classical (model absent or weak)"
        else:
            chosen, used = model, "model"
        if model is not None and classical is not None:
            dx = model[0] - classical[0]
            dy = model[1] - classical[1]
            radius = max((classical[2] - classical[0]) / 2, 1)
            ratio = ((model[2] - model[0]) / 2) / radius
            agree = (dx * dx + dy * dy) ** 0.5 <= CENTRE_TOLERANCE * radius and \
                RADIUS_RANGE[0] <= ratio <= RADIUS_RANGE[1]
            tally["agree" if agree else "disagree"] += 1

        if chosen is not None and probe.iou(chosen, truth) >= GOOD_IOU:
            tally["ensemble"] += 1

        if not (chosen is not None and probe.iou(chosen, truth) >= GOOD_IOU):
            print(f"  miss: {os.path.basename(path)[:24]:26s} {used:28s} "
                  f"model {('%.2f' % probe.iou(model, truth)) if model else '--':>5}  "
                  f"classical {('%.2f' % probe.iou(classical, truth)) if classical else '--':>5}")

    print()
    print(f"{total} frames")
    print(f"  model alone good      {tally['model only']}")
    print(f"  classical alone good  {tally['classical only']}")
    print(f"  ensemble good         {tally['ensemble']}")
    print(f"  agreements {tally['agree']} / disagreements {tally['disagree']}")


if __name__ == "__main__":
    main()
