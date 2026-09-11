"""Train the cabinet-screen detector and export it for the app.

The app needs CoreML, which needs a Mac, so this script stops at ONNX: Codemagic
converts the ONNX to .mlpackage during the iOS build. Nothing here is required
to reproduce the app other than the resulting .onnx file.

Run with a Python that has torch:
    D:\\Python312\\python.exe tools\\train_yolo.py

Output:
    C:\\Users\\93543\\fisheye_train\\run\\weights\\best.pt
    C:\\Users\\93543\\fisheye_train\\run\\weights\\best.onnx   <- commit this
"""

import os
import shutil
import sys

DATASET = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_yolo", "data.yaml")
PROJECT = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_train")
RUN = "run"

EPOCHS = 120
IMGSZ = 640
BATCH = 16
DEVICE = 0            # RTX 3060; use "cpu" if CUDA is unavailable


def main():
    if not os.path.exists(DATASET):
        print("run tools/make_yolo_dataset.py first, missing", DATASET)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=2,               # Windows dataloader is happier with a small pool
        project=PROJECT,
        name=RUN,
        exist_ok=True,
        seed=0,
        patience=40,
        # A circle is robust to flip and modest rotation; the frame is a fisheye
        # so scale jitter matters (the screen changes size with distance).
        degrees=10.0,
        translate=0.10,
        scale=0.40,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=20,
        plots=True,
    )

    best = os.path.join(PROJECT, RUN, "weights", "best.pt")
    if not os.path.exists(best):
        print("no best.pt produced")
        sys.exit(1)

    exported = YOLO(best).export(format="onnx", imgsz=IMGSZ, opset=12,
                                 dynamic=False, simplify=False, half=False)
    target = os.path.join(PROJECT, RUN, "weights", "best.onnx")
    if os.path.abspath(exported) != os.path.abspath(target):
        shutil.copyfile(exported, target)

    print()
    print("ONNX:", target, f"({os.path.getsize(target) / 1e6:.2f} MB)")

    # A quick numeric check on the val split, so a broken export is caught here
    # rather than at CoreML conversion time on Codemagic.
    metrics = YOLO(best).val(data=DATASET, imgsz=IMGSZ, device=DEVICE, verbose=False)
    print(f"mAP50 {metrics.box.map50:.3f}   mAP50-95 {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
