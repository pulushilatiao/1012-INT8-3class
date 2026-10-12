from ultralytics import YOLO

model = YOLO("best.pt")

print("Task:", model.task)
print("Classes:", model.names)
print("Number of classes:", len(model.names))