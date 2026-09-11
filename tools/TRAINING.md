# 训练机台检测模型（YOLO → ONNX → Core ML）

给「对准机台 / 锁定构图」用的模型流水线。**整条链已经跑通一遍**，下面是实际用的命令和实测结果。

---

## 为什么走模型

经典 CV 靠"屏幕圆边界 + 边缘验证"，在已有素材上表现很好（50 张里预标注基本可直接用），
但它是**手工设计的判据**：换机台型号、换灯光、屏幕被大面积遮挡时，需要重新调参数。

YOLO 学的是"机台长什么样"，不依赖预设规则。代价是需要标注数据和一次训练。

---

## ① 标注

```bash
python tools/label_tool.py <素材文件夹> [<更多文件夹>]
```

工具会先用经典检测器画好绿框，绝大多数直接空格接受。**全部操作都能用键盘完成**
（鼠标拖动是坐标被缩放过一次就很难对齐，只作备用）：

| 按键 | 作用 |
|---|---|
| `空格` / `回车` | **接受当前框，下一张**（主力键） |
| `方向键` / `WASD` | 移动框（按住 Shift 大写 = 5 倍步长） |
| `-` / `=` | 缩小 / 放大框 |
| `c` | 换下一个候选圆（检测器给 6 个） |
| `r` | 全分辨率重新检测 |
| `n` / `u` / `q` | 跳过 / 退回重标 / 保存退出 |

框的颜色：**绿 = 检测器提议**，**青 = 你手调过**。

**框住的是「内屏那个圆」**，不含外键、不含外壳 —— 圆心就是机台中心，半径是稳定尺子。

**断点续标**：已经标过的帧会自动跳过，重开工具从断的地方继续。

输出（**刻意放在仓库外**，不会误提交）：

```
%USERPROFILE%\fisheye_dataset\
    images\*.jpg
    labels\*.txt        ← YOLO 格式: 0 cx cy w h（全部归一化）
```

---

## ② 切数据集

```bash
python tools/make_yolo_dataset.py
```

按固定规则切 train/val（每 5 张留 1 张验证），同时当质检：

- 标签坐标越界会报出来
- **框的宽高比**中位数应该 ≈ 1.00（目标是圆）；离 1.0 太远的说明标歪了

输出 `%USERPROFILE%\fisheye_yolo\`。

> 素材量：现有 50 张只是**练手**（它们是预览画面的截图，不是原始鱼眼帧）。
> 真正要用的训练集必须来自 app 的 `校正面板 → 保存原始帧` —— 那才是推理时看到的同一分布。
> 建议 100~300 张，覆盖不同歌曲的按键颜色、手遮挡、多人围观。

---

## ③ 训练

torch/ultralytics 需要 Python ≤3.13，本机用 `D:\Python312`：

```powershell
D:\Python312\python.exe -m pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
D:\Python312\python.exe tools\train_yolo.py
```

`tools/train_yolo.py` 里写死了这套参数（YOLOv8n / 640 / batch16 / 120 轮 / patience 40），
并且训练完**自动导出 ONNX 并跑一次 val**：

| 实测（50 张，40 训练 / 10 验证） | 值 |
|---|---|
| best epoch | 16（patience 40 提前停） |
| mAP50 / mAP50-95 | **0.995 / 0.821** |
| 训练耗时 | 56 轮 ≈ 1 分钟（RTX 3060） |
| 模型 | YOLOv8n，3.0M 参数，8.1 GFLOPs |

---

## ④ ONNX 导出与校验

`train_yolo.py` 结尾自动做了，等价命令：

```powershell
yolo export model=best.pt format=onnx imgsz=640 opset=12 simplify=False
# → models\screen.onnx  输入 images [1,3,640,640]，输出 output0 [1,5,8400]
```

**导出后必须校验**（`tools/check_onnx.py`）—— 这一步救过一次：

```powershell
D:\Python312\python.exe tools\check_onnx.py
```

它扫描输入尺度 / 通道序 / 裁剪方式，报出解码框和标注框的 IoU：

| 输入 | 结果 |
|---|---|
| **0–1（×1/255）, RGB, centerCrop** | **IoU 0.88** ✅ 正确配方 |
| 0–255（多除了一次） | IoU 0.18，而且**10/10 都"很有信心"** ❌ |
| 0–1, BGR | IoU 0.86（能用，但不是训练时的约定） |

两个必须记住的坑：

1. **输入必须是 0–1**。喂 0–255 会得到**自信但完全错误的框**——看起来"检测到了"，其实全是废话。
   Core ML 转换时对应 `ct.ImageType(scale=1/255.0)`。
2. **这个模型的分数天生很低**：正确框只有 0.13~0.41。用 0.5 当阈值会一直"检测不到"，
   而错误的饱和输入反而更"自信"。所以阈值取 **0.08**，动作门槛 **0.15**。

> 校验用的参照系是 ultralytics 自己的 ONNX 推理：同样这些帧，它给出的 IoU 是 0.82~0.95。
> 只有自己的解码和它对齐了，才能说明布局（channels-first，`(1, 5, 8400)`）没错。

---

## ⑤ Core ML 转换（构建时自动完成）

`coremltools` 只有 macOS 版 → 转换放在 Codemagic 构建里做，产物**不进仓库**：

```yaml
- name: Convert the trained detector to CoreML
  script: |
    python3 -m venv "$HOME/mlvenv"
    "$HOME/mlvenv/bin/pip" install -q coremltools onnx
    "$HOME/mlvenv/bin/python" tools/onnx_to_coreml.py
```

`tools/onnx_to_coreml.py` 写出 `SteadyFisheye/ScreenDetector.mlpackage`，
Xcode 在 Resources 阶段把它编成 `ScreenDetector.mlmodelc` 塞进 bundle。

app 端 `ScreenDetector.swift` 用 Vision 跑：

- **模型在** → `VNCoreMLRequest`，`.centerCrop`（和训练时的方形裁切一致）
- **模型不在 / 分数 < 0.15** → **自动回退到经典圆检测器**
- 检测结果统一成「圆心 + 半径」，两趟对准逻辑完全不用改

模型训练失败、转换失败、删掉模型文件 —— app 都照常能用，只是退回经典检测器。

---

## 当前状态

| 步骤 | 状态 |
|---|---|
| ① 标注工具（键盘操作 + 断点续标） | ✅ 完成，50 张已标 |
| ② 数据集切分 + 质检 | ✅ 完成（`tools/make_yolo_dataset.py`） |
| ③ 训练 | ✅ 跑通（mAP50 0.995，练手集） |
| ④ ONNX 导出 + 校验 | ✅ 完成，配方已实测（`tools/check_onnx.py`） |
| ⑤ Core ML 集成 + 回退 | ✅ 完成（`ScreenDetector.swift` + `codemagic.yaml`） |
| 真素材训练集 | ⏳ 等 `保存原始帧` 攒够 100+ 张再训一轮 |

**下一轮要做的**：用原始帧重训 → 覆盖 `models/screen.onnx` → 推上去，
Codemagic 自动重新转换打包，app 不用改一行代码。
