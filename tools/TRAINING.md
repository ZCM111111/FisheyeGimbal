# 训练机台检测模型（YOLO → Core ML）

给「对准机台」用的模型流水线。**整个过程分五步，只有第一步需要你动手。**

---

## 为什么走模型

经典 CV 靠"屏幕圆边界 + 边缘验证"，在已有素材上表现很好（62 张里 61 张认对），
但它是**手工设计的判据**：换机台型号、换灯光、屏幕被大面积遮挡时，需要我重新调参数。

YOLO 的优势是**抗干扰**——它学的是"机台长什么样"，不依赖我预设的规则。
代价是需要标注数据和一次训练。

---

## ① 标注（约 3~5 分钟）

```bash
python tools/label_tool.py <素材文件夹>
```

**工具会用我的经典检测器先画好框**（绿框），实测**62 张里 61 张预标注可直接用**，你只需要：

| 操作 | 说明 |
|---|---|
| `空格` / `回车` | **接受绿框**（绝大多数情况） |
| 按住鼠标拖 | 框不对时重画 |
| `r` | 重新检测这一帧 |
| `n` | 跳过这张 |
| `u` | 退回上一张 |
| `q` | 保存退出 |

**框住的是「内屏的圆形区域」**，不是整个机台、也不包括外键。

输出（**刻意放在仓库外**，不会误提交）：

```
%USERPROFILE%\fisheye_dataset\
    images\*.jpg
    labels\*.txt        ← YOLO 格式: 0 cx cy w h（全部归一化）
    data.yaml
```

> 想换位置：`python tools/label_tool.py <素材文件夹> --out D:\某处`

---

## ② 数据量建议

| 素材来源 | 目标 |
|---|---|
| 现有 62 张 | 先跑一轮，看效果 |
| **街机厅实拍**（最重要） | 越多越好，建议 100~300 张 |
| 各种歌曲/按键颜色 | 每首歌颜色不同，多覆盖几种 |
| 手遮挡、多人挤在机台前 | 各来一些 |

**采集方式**：app 里 `校正面板 → 镜头标定 → 保存原始帧`，一次存一张。
**加进数据集**：把它们放进素材文件夹，重跑标注工具，已经标过的会自动带框跳过。

---

## ③ 训练（需要 Python 环境，CPU 也能跑）

```bash
pip install ultralytics

# 单类检测，nano 版（6MB，手机端实时够用）
yolo detect train data=%USERPROFILE%\fisheye_dataset\data.yaml ^
     model=yolov8n.pt epochs=100 imgsz=640 batch=8

# 看结果
yolo detect predict model=runs\detect\train\weights\best.pt ^
     source=<素材文件夹> save=True
```

**判断训练是否成功**：`runs/detect/train/` 里的 `mAP50` 到 0.9 以上基本就够用。
如果只有 0.6 左右 → 加数据（第②步）。

---

## ④ 导出 ONNX

```bash
yolo export model=runs\detect\train\weights\best.pt format=onnx imgsz=640
# 得到 best.onnx
```

**为什么不直接导出 Core ML**：`coremltools` 只能在 macOS 上运行，
你的构建机是 Codemagic（macOS）→ **转换放到构建时做** ✓

把 `best.onnx` 提交到仓库的 `models/` 目录（这文件是你的模型，不涉及素材）。

---

## ⑤ Core ML 转换（构建时自动完成）

`codemagic.yaml` 里加一步：ONNX → Core ML → 塞进 app bundle。

```yaml
- name: Convert detector model
  script: |
    pip3 install coremltools onnx
    python3 tools/onnx_to_coreml.py models/best.onnx models/CabinetScreen.mlmodel
```

app 端用 Vision 加载（`CabinetModelDetector.swift`）：

- **模型存在** → 用模型，输出屏幕 bbox → 圆心 = bbox 中心
- **模型不存在** → 回退到经典检测器（现在的版本）

这样即使模型训练失败，app 照样能用 ✓

---

## 当前状态

| 步骤 | 状态 |
|---|---|
| ① 标注工具 | ✅ 完成（`tools/label_tool.py`）|
| ② 数据集积累 | ⏳ 需要你采集 |
| ③ 训练 | ⏳ 需要你跑 |
| ④ ONNX 导出 | ⏳ 需要你跑 |
| ⑤ Core ML 集成 + 回退 | ⏳ 等你给出 `best.onnx` 我就接 |

**在模型就绪之前，app 里的经典检测器继续用**——它的独立价值是：预标注工具、以及模型失败时的回退。
