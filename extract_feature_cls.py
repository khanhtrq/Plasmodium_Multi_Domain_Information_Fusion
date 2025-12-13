import argparse
import os
import cv2
import numpy as np
import json
import torch
from mmpretrain.apis import FeatureExtractor
from pathlib import Path
from utils.confusion_matrix import DetectionConfusionMatrix


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification feature extraction")

# Add arguments

parser.add_argument("--cls_model", type=str, help="path to config file")
parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")

parser.add_argument("--gt_folder", type=str, help="folder with annotation (entire pipeline)")

parser.add_argument('--conf_threshold', type=float, default=0.3, help="Confidence threshold for the whole pipeline")
parser.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for the whole pipeline")

parser.add_argument("--extraction_folder", type=str, help= "Directory to save confusion matrix.")

parser.add_argument("--save_dir", type=str, help= "Directory to save confusion matrix.")
parser.add_argument("--cls_batch_size", type=int, default=32, help="batch size")

parser.add_argument('--num_classes', type=int, default= 7, help="Number of classes for confusion matrix")

parser.add_argument("--annotation_file", type=str)
parser.add_argument("--data_root", type=str)



args = parser.parse_args()

# ---------------
# CLASSIFICATION
# ---------------

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
txt_something = os.path.join(args.extraction_folder, "labels")

extracted_features = extractor(inputs = image_names,
                            batch_size=args.cls_batch_size)

print("Type of feature:", type(extracted_features))
print("Type of feature [0]:", type(extracted_features[0]))
print("Length of feature [0]:", len(extracted_features[0]))
print("Type of feature [0][0]:", type(extracted_features[0][0]))


for feature in extracted_features:
    print(extracted_features[0][0].shape)
    feature_list.append(feature[0].cpu().numpy())
    break

print("Length of labels:", len(labels))
print("Length of extracted_features:", len(feature_list))

exit()

for rbc_folder in os.listdir(args.extraction_folder):

    folder_path = os.path.join(args.extraction_folder, rbc_folder)
    input_images = []
    for root, _, files in os.walk(folder_path):        
        for file in files:
            if file.lower().endswith(".jpg"):
                input_images.append(os.path.abspath(os.path.join(root, file)))

    extracted_features = extractor(inputs = input_images,
                                batch_size=args.cls_batch_size)
    
    print("RBC folder:", rbc_folder)
    print("Type of feature:", type(extracted_features))
    print("Type of feature [0]:", type(extracted_features[0]))
    print("Length of feature [0]:", len(extracted_features[0]))
    print("Type of feature [0][0]:", type(extracted_features[0][0]))


    for feature in extracted_features:
        print(extracted_features[0][0].shape)
        break
    
    '''
    txt_file = [f for f in os.listdir(os.path.join(detection_save_dir, "labels")) if f.startswith(rbc_folder)][0]
    cell_detection_result_file = os.path.join(txt_result_dir, txt_file)
    
    os.makedirs(os.path.join(args.save_dir, "life_cycle_labels"), exist_ok=True)
    cls_detection_result = os.path.join(os.path.join(args.save_dir, "life_cycle_labels"), txt_file)
    cls_detection_result_list = []

    # detection result, format: Saves detection resut
    # format: [class] [x_center] [y_center] [width] [height] [confidence] 
    with open(cell_detection_result_file, "r") as file:
        lines = file.readlines()

    #list of predictions to compute confusion matrix
    pred_conf = []
    image_path = []
    
    # format: [class] [x_center] [y_center] [width] [height] [confidence] 
    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_name, x_center, y_center, w, h, conf_score = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)

        # <class_name> <confidence> <left> <top> <right> <bottom>      
        x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
        x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
        # cls_class_name = classification_results[i]['pred_label']
        # pred_score = classification_results[i]['pred_score']

        cls_class_name = classification_results[i].pred_label
        pred_score = classification_results[i].pred_score
        image_path.append(classification_results[i].metainfo['filename'])

        if args.merge_healthy_other:
            # if label == other then label = healthy
            if cls_class_name == 5:
                cls_class_name = 4
        
        cls_detection_result_list.append('{} {} {} {} {} {}\n'.format(cls_class_name, conf_score, x1, y1, x2, y2))

        #predictions results to compute confusion matrix
        pred_conf.append([x1, y1, x2, y2, conf_score, cls_class_name])
    pred_conf = np.array(pred_conf, dtype = object)

    #Grounth truth, format <label>, <x1>, <y1>, <x2>, <y2>
    #list of grounth truth to compute confusion matrix

    gt_path = os.path.join(args.gt_folder, txt_file)
    gt_conf = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    for gt_sample in lines:
        gt_label, x1, y1, x2, y2 = gt_sample.split(' ')
        gt_label = int(gt_label)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gt_conf.append([gt_label, x1, y1, x2, y2])
    gt_conf = np.array(gt_conf, dtype = object)


    #Save the refined result
    with open(cls_detection_result, "w") as refined_file:
        for line in cls_detection_result_list:
           refined_file.write(line)

    '''

exit()

detection_conf = detection_conf_obj.return_matrix()
image_path = detection_conf_obj.return_image_path()
print(image_path)
with open(os.path.join(args.save_dir, 'iamge_path.json'), 'w') as f:
    json.dump(image_path, f, indent=4)

pr_metrics = detection_conf_obj.compute_PR_from_matrix(detection_conf)

print("Whole pipeline confusion matrix:")
print(torch.tensor(detection_conf, dtype=torch.int64))
print("Precision Recall:", pr_metrics)

#Save confusion matrix
os.makedirs(args.save_dir, exist_ok= True)
data = pr_metrics
data["confusion_matrix"] = detection_conf.tolist()
with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
    json.dump(data, f, indent=4)
