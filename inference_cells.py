from mmpretrain.apis import ImageClassificationInferencer
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--model", type=str, help="path to config file")
parser.add_argument("--pretrained", type=str, help="path to model pretrained (trained parameteres) pt file")
parser.add_argument("--inputs", type=str, help="path to images folder")

args = parser.parse_args()

inferencer = ImageClassificationInferencer(
    model = args.model,
    pretrained = args.pretrained)

inferencer(inputs = args.inputs,
          show_dir = './visualize/')