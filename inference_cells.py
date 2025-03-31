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
    input_images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            input_images.append(os.path.abspath(os.path.join(root, file)))

inferencer = ImageClassificationInferencer(
    model = args.model,
    pretrained = args.pretrained,
    device='cuda')

classification_results = inferencer(inputs = input_images,
                                    show_dir = '../visualize/')

# [{'pred_scores': array([9.8446275e-05, 9.1670381e-06, 1.7500604e-06, 5.8469166e-05,
#          9.3389821e-01, 6.5715335e-02, 2.1869126e-04], dtype=float32),
#   'pred_label': 4,
#   'pred_score': 0.9338982105255127},
#  {'pred_scores': array([9.8446275e-05, 9.1670381e-06, 1.7500604e-06, 5.8469166e-05,
#          9.3389821e-01, 6.5715335e-02, 2.1869126e-04], dtype=float32),
#   'pred_label': 4,
#   'pred_score': 0.9338982105255127}]