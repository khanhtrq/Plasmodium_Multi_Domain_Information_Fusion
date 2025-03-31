from mmpretrain.apis import ImageClassificationInferencer
import argparse
from pathlib import Path
import os


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--model", type=str, help="path to config file")
parser.add_argument("--pretrained", type=str, help="path to model pretrained (trained parameteres) pt file")
parser.add_argument("--inputs", type=str, help="path to image or images folder")

args = parser.parse_args()


if Path(args.inputs).is_file():
    input_images = [args.inputs]
else:
    folder_path = args.inputs
    input_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    input_images = [
        os.path.abspath(os.path.join(folder_path, f)) 
        for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
    ]

inferencer = ImageClassificationInferencer(
    model = args.model,
    pretrained = args.pretrained,
    device='cuda')

inferencer(inputs = input_images,
          show_dir = '../visualize/')