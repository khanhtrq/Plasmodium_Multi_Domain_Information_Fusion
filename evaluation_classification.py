from mmpretrain.apis import ImageClassificationInferencer
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import json

from mmengine.config import Config

from mmpretrain.evaluation.metrics.single_label import ConfusionMatrix, Accuracy
from mmpretrain.evaluation.metrics.minority_metrics import MinorityMetrics

# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--cls_model", type=str, help="path to config file")
parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")
parser.add_argument("--annot_file", type=str)
parser.add_argument('--num_classes', type=int, default= 7, help="Number of class for confusion matrix")
parser.add_argument("--data_root", type=str, help="path to data")


parser.add_argument("--cls_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--saved_path", type=str, default="evaluation_results")


args = parser.parse_args()

config = Config.fromfile(args.cls_model)

data_root = args.data_root
batch_size = config.batch_size

# print(config.data_root)
# print(config.batch_size)

with open(args.annot_file, 'r') as f:
    lines = f.readlines()

# print(lines[0])

img_paths = []
gt_labels = np.zeros(len(lines), dtype=int)

for i, line in enumerate(lines):
    img_path, gt_label = line.split(' ')
    img_paths.append(os.path.join(data_root, img_path))
    gt_labels[i] = int(gt_label)

inferencer = ImageClassificationInferencer(
    model = args.cls_model,
    pretrained = args.cls_pretrained,
    device= 'cuda')

cls_results = inferencer(inputs = img_paths,
                        show_dir = './visualize/',
                        batch_size=args.cls_batch_size)

pred_labels = np.zeros(len(cls_results), dtype=int)
for i in range(len(cls_results)):
    pred_labels[i] = cls_results[i]['pred_label']


confusion_matrix = ConfusionMatrix.calculate(pred= pred_labels, target= gt_labels, num_classes= args.num_classes)
acc = Accuracy.calculate(pred=pred_labels, target=gt_labels)
class_metrics = MinorityMetrics.compute_metrics_from_matrix(confusion_matrix=confusion_matrix,
                                                               num_classes= args.num_classes)

print(confusion_matrix)
print(acc.item())
print(class_metrics)

evaluation_result = {
    "accuracy": acc.item(),
    "class_metrics": class_metrics,
    "confusion_matrix": confusion_matrix.tolist()
}

os.makedirs(args.saved_path, exist_ok=True)
with open(os.path.join(args.saved_path, 'evaluation_result.json'), 'w') as f:
    json.dump(evaluation_result, f, indent=4)