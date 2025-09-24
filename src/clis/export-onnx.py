from ultralytics import YOLO
from argparse import ArgumentParser

def export_onnx(model_path: str, output_path: str):
    # Load the YOLO model
    model = YOLO(model_path)

    # Export the model to ONNX format
    model.export(format='onnx')
    print(f'Model exported to {output_path}')

if __name__ == '__main__':
    parser = ArgumentParser(description='Export YOLO model to ONNX format')

    parser.add_argument(
        'model_path', type=str, help='Path to the YOLO model file (e.g., yolo11n.pt)'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Output path for the ONNX model (e.g., model.onnx)',
    )

    args = parser.parse_args()
    export_onnx(args.model_path, args.output_path)
