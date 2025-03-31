from ultralytics import YOLO
import argparse


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--detection_model", type=str, help="detection_model")
parser.add_argument("--detection_inputs", type=str, help="path to images folder")

args = parser.parse_args()

detection_model = YOLO(args.detection_model)

detection_results = detection_model.predict(source=args.detection_inputs)