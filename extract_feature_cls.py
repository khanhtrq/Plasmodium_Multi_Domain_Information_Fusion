import argparse
import os
import cv2
import numpy as np
import json
import torch
from mmpretrain.apis import FeatureExtractor, ImageClassificationInferencer
from pathlib import Path
from utils.confusion_matrix import DetectionConfusionMatrix


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification feature extraction")

# Add arguments

parser.add_argument("--cls_model", type=str, help="path to config file")
parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")
parser.add_argument("--save_dir", type=str, help= "Directory to save confusion matrix.")
parser.add_argument("--cls_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--annotation_file", type=str)
parser.add_argument("--data_root", type=str)


args = parser.parse_args()

data_root = args.data_root
image_names = []
labels = []
feature_list = []

with open(args.annotation_file, "r") as file:
    lines = file.readlines()
for i, line in enumerate(lines):
    parts = line.strip().split()
    image_file_i, cls_idx_i = parts[0], int(parts[1])

    image_names.append(os.path.join(data_root, image_file_i))
    labels.append(cls_idx_i)

extractor = FeatureExtractor(
    model = args.cls_model,
    pretrained = args.cls_pretrained,
    device='cuda')

extracted_features = extractor(inputs = image_names,
                            batch_size=args.cls_batch_size,
                            mode = 'predict',
                            # stage = 'neck'
                            )

print("Type of feature:", type(extracted_features))
print("Type of feature [0]:", type(extracted_features[0]))
print("Length of feature [0]:", len(extracted_features[0]))
print("Type of feature [0][0]:", type(extracted_features[0][0]))


for feature in extracted_features:
    feature_list.append(feature[0].cpu().numpy())

print("Length of labels:", len(labels))
print("Length of extracted_features:", len(feature_list))

feature_arr = np.array(feature_list)
labels = np.array(labels)
print(feature_arr.shape)
print(labels.shape)

print(f"Feature max:{feature_arr.max()}, min:{feature_arr.min()}, mean: {feature_arr.mean()}")
np.save(os.path.join(args.save_dir, "features.npy"), feature_arr)
np.save(os.path.join(args.save_dir, "labels.npy"), labels)

exit()