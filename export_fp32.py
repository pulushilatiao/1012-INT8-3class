##  (yolo_env) hsp@ihpc-4090-2:~/20261012/SplitDataset$ sha256sum 1012fp32.onnx
##fbdac4c36766116b65594a66fb8651b9befc6bc5f4a8677c58bf8362c8c38211  1012fp32.onnx


# export_tomato_fp32.py
from pathlib import Path
from ultralytics import YOLO
import onnx
import shutil

# 修改为你的正确权重绝对路径
WEIGHTS = "best1115yolov8n.pt"

OUTPUT = Path("1012fp32.onnx").resolve()

model = YOLO(WEIGHTS)

assert model.task == "detect"
assert list(model.names.values()) == [
    "tomato", "leaf", "stem"
]

print("Verified custom-trained model:", model.names)

export_path = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    opset=13,
    simplify=False,
    nms=False
)

export_path = Path(export_path).resolve()

if export_path != OUTPUT:
    shutil.copy2(export_path, OUTPUT)

m = onnx.load(str(OUTPUT))
onnx.checker.check_model(m)

shape = [
    d.dim_value if d.HasField("dim_value")
    else d.dim_param
    for d in m.graph.output[0]
    .type.tensor_type.shape.dim
]

print("ONNX output:", shape)

assert shape == [1, 7, 8400], (
    "Unexpected ONNX output: {}".format(shape)
)

print("PASS: Three-class ONNX exported")
print("File:", OUTPUT)
