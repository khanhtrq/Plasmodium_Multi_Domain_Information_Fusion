from mmpretrain.apis import ImageClassificationInferencer
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np



# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--cls_model", type=str, help="path to config file")
parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")
parser.add_argument("--rbc_images", type=str, help="path to images folder")
parser.add_argument("--detection_save_dir", type=str, help="path to image or images folder")

parser.add_argument("--cls_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--saved_name", type=str, default="val_prediction_score")


args = parser.parse_args()

inferencer = ImageClassificationInferencer(
    model = args.cls_model,
    pretrained = args.cls_pretrained,
    device='cuda')
txt_result_dir = os.path.join(args.detection_save_dir, "labels")

image_path = []
pred_labels = [] #list of prediction label
pred_scores = [] #list of np array for prediction score
score_1st = [] #list of highest prediction scores
score_2nd = [] #list of second highest prediction scores
gt_class_names = [] #list of class names
pred_score_difference = []

inference_data = {"image_path": image_path,
                  "gt_class_name": gt_class_names,
                  "pred_label": pred_labels,
                  "pred_scores": pred_scores,
                  "score_1st": score_1st,
                  "score_2nd": score_2nd,
                  "score_difference": pred_score_difference}

input_images = []
for root, _, files in os.walk(args.rbc_images):
    for file in files:
        if file.split('.')[-1] == 'jpg':
            input_images.append(os.path.abspath(os.path.join(root, file)))
            gt_class_names.append(os.path.split(root)[-1])

            image_path.append(os.path.abspath(os.path.join(root, file)))
            
    

classification_results = inferencer(inputs = input_images,
                                    show_dir = './visualize/',
                                    batch_size=args.cls_batch_size)

for instance_pred in classification_results:    
    sorted_scores = np.sort(instance_pred['pred_scores'])
    sorted_scores = np.flip(sorted_scores)
    
    score_1st.append(sorted_scores[0])
    score_2nd.append(sorted_scores[1])
    pred_score_difference.append(sorted_scores[0] - sorted_scores[1])

    pred_labels.append(instance_pred['pred_label'])
    pred_scores.append(instance_pred['pred_scores'])

inference_df = pd.DataFrame(inference_data)

inference_df.to_csv("{}.csv".format(args.saved_name))
inference_df.to_excel("{}.xlsx".format(args.saved_name))