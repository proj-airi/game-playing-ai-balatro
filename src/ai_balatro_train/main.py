from ultralytics import YOLO

# Load a pretrained model (recommended for training)
model = YOLO('yolo11n.pt')

# Train the model with your Balatro dataset
results = model.train(
    data='configs/v1-balatro-ui/dataset.yaml',
    epochs=2000,
    imgsz=640,
    batch=64,
    project='runs',
    name='v1-balatro-ui-2000-epoch',
)
