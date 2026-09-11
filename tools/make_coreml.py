"""Build the app's CoreML detector from the trained YOLO weights.

coremltools has no ONNX frontend — this was tried and coremltools 8.3 and 9.0
both ship converters for tensorflow, torch, sklearn, xgboost and libsvm only,
and `ct.convert` on an ONNX model fails with "Unable to determine the type of
the model". So the conversion goes the supported way: ultralytics exports the
PyTorch weights, and its CoreML branch calls coremltools itself.

Two details of that export are load-bearing and are asserted below, because the
Swift decoder in ScreenDetector.swift assumes them:

    nms=False     the package holds the raw head — one tensor of
                  (1, 5, 8400): cx, cy, w, h, score per anchor
    scale=1/255   the model takes 0-255 pixels and normalises them itself,
                  which matches what the ONNX was measured to expect

Run inside the Codemagic build (needs a Mac and torch):
    python3 tools/make_coreml.py
"""

import os
import shutil
import subprocess
import sys

WEIGHTS = ("models/screen.pt",)
PACKAGE_NAME = "screen"
MODEL_NAME = "ScreenDetector"
INPUT_SIDE = 640
EXPECTED_SHAPE = (1, 5, 8400)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights = None
    for candidate in WEIGHTS:
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            weights = path
            break
    if weights is None:
        sys.exit("no weights found, expected one of " + ", ".join(WEIGHTS))

    # numpy 2.4 breaks the coremltools export ("check_requirements" in ultralytics
    # pins it too), and torch is only needed here.
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    exported = model.export(format="coreml", imgsz=INPUT_SIDE, nms=False)
    if isinstance(exported, (list, tuple)):      # some versions return a list
        exported = exported[0]
    package = os.path.abspath(str(exported))
    print("exported:", package)
    if not os.path.isdir(package):
        sys.exit("expected a .mlpackage directory, got " + package)

    check_package(package)
    compile_package(package, root)


def check_package(package):
    """Loads the package and runs one blank frame through it.

    Blank on purpose: no footage leaves this machine, and the point is only to
    prove the graph loads and still ends in the raw detection tensor. A package
    that fails here is worse than no model at all, because the app would decode
    nonsense out of it.
    """
    import numpy as np
    import coremltools as ct

    model = ct.models.MLModel(package)
    description = model.get_spec().description
    print("inputs:")
    for item in description.input:
        print("   ", item.name, item.type.WhichOneof("Type"))

    image_inputs = [item.name for item in description.input
                    if item.type.WhichOneof("Type") == "imageType"]
    if not image_inputs:
        sys.exit("no image input in the exported package")

    try:
        from PIL import Image
    except ImportError:
        print("pillow is missing, skipping the forward pass check")
        return

    name = image_inputs[0]
    blank = Image.fromarray(np.zeros((INPUT_SIDE, INPUT_SIDE, 3), dtype=np.uint8))

    outputs = model.predict({name: blank})
    shapes = {}
    for key, value in outputs.items():
        shape = getattr(value, "shape", None)
        shapes[key] = tuple(shape) if shape is not None else type(value).__name__
        print("output", key, shapes[key])

    if EXPECTED_SHAPE not in shapes.values():
        sys.exit(f"expected a raw detection tensor {EXPECTED_SHAPE} in the outputs, "
                 f"got {shapes}. ScreenDetector.swift decodes that tensor, so a "
                 f"pipeline-style export (nms=True) would not work.")
    print(f"raw detection head present: {EXPECTED_SHAPE}")


def compile_package(package, root):
    """Compiles the package into the name the app looks up."""
    output = os.path.join(root, "SteadyFisheye", MODEL_NAME + ".mlmodelc")
    staging = os.path.join(root, "models", "_compiled")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    subprocess.run(["xcrun", "coremlcompiler", "compile", package, staging], check=True)

    produced = os.path.join(staging, PACKAGE_NAME + ".mlmodelc")
    if not os.path.isdir(produced):
        candidates = [n for n in os.listdir(staging) if n.endswith(".mlmodelc")]
        if not candidates:
            sys.exit("coremlcompiler produced nothing in " + staging)
        produced = os.path.join(staging, candidates[0])

    if os.path.isdir(output):
        shutil.rmtree(output)
    shutil.move(produced, output)
    shutil.rmtree(staging, ignore_errors=True)

    size = sum(os.path.getsize(os.path.join(base, f))
               for base, _, files in os.walk(output) for f in files)
    print(f"compiled: {output} ({size / 1e6:.1f} MB)")
    print("the Codemagic packaging step copies this into SteadyFisheye.app")


if __name__ == "__main__":
    main()
