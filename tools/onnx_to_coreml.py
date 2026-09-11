"""Convert the trained YOLOv8 ONNX detector into a CoreML package for the app.

coremltools only runs on macOS, so this is meant to run inside the Codemagic
build rather than on the machine the model was trained on. The app ships the
resulting .mlpackage compiled into its bundle, and falls back to the classical
circle detector when the package is absent.

The preprocessing declared here has to match what the ONNX was exported for, and
that was measured, not assumed — see tools/check_onnx.py:

    input scale   1/255   (the graph expects 0-1; feeding it 0-255 produces
                           confident, wrong boxes on every frame)
    colour layout RGB     (the training loader was RGB; BGR still detects but
                           with a lower IoU)
    geometry      the app asks Vision for .centerCrop, so the model sees a
                  square centre crop, which is what it was trained on

Usage (on macOS):
    python3 tools/onnx_to_coreml.py
"""

import os
import shutil
import sys

MODEL_NAME = "ScreenDetector"
INPUT_SIDE = 640


def main():
    try:
        import coremltools as ct
        import onnx
        import numpy as np  # noqa: F401  (coremltools needs it)
    except ImportError as error:
        sys.exit(f"missing conversion dependency: {error}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(root, "models", "screen.onnx")
    out_path = os.path.join(root, "SteadyFisheye", MODEL_NAME + ".mlpackage")
    if not os.path.exists(onnx_path):
        sys.exit("missing " + onnx_path)

    graph = onnx.load(onnx_path)
    input_name = graph.graph.input[0].name
    output_name = graph.graph.output[0].name
    dims = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    print(f"onnx input {input_name} {dims}, output {output_name}")
    if dims != [1, 3, INPUT_SIDE, INPUT_SIDE]:
        print(f"unexpected input shape {dims}; the app assumes "
              f"[1, 3, {INPUT_SIDE}, {INPUT_SIDE}]")

    model = ct.convert(
        onnx_path,
        inputs=[ct.ImageType(name=input_name,
                             shape=(1, 3, INPUT_SIDE, INPUT_SIDE),
                             scale=1.0 / 255.0,
                             bias=[0.0, 0.0, 0.0],
                             color_layout=ct.colorlayout.RGB)],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    model.short_description = "maimai cabinet screen (YOLOv8n, one class)"
    model.input_description[input_name] = "portrait capture frame"
    model.output_description[output_name] = "cx, cy, w, h, score for 8400 anchors"

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    model.save(out_path)

    # The bundle needs the compiled form, and Vision loads it by name, so the
    # package name is what ScreenDetector.swift looks up.
    size = sum(os.path.getsize(os.path.join(base, name))
               for base, _, names in os.walk(out_path) for name in names)
    print(f"wrote {out_path} ({size / 1e6:.1f} MB)")
    print("the Xcode build compiles it into ScreenDetector.mlmodelc in the app bundle")


if __name__ == "__main__":
    main()
