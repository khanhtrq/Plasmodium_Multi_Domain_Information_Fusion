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
parser.add_argument("--detection_save_dir", type=str, help="path to image or images folder")


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

#----------
# DETECTION RSULT
# ---------

txt_result_dir = os.path.join(args.detection_save_dir, "labels")
txt_file_list = [f for f in os.listdir(txt_result_dir) if os.path.isfile(os.path.join(txt_result_dir, f))]

for txt_file in txt_file_list:
    result_file = os.path.join(txt_result_dir, txt_file)
    
    refined_result = os.path.join(txt_result_dir, txt_file[:-3] + '_refined.txt')
    refined_result_list = []

    with open(result_file, "r") as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_name, x_center, y_center, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        refined_class_name = classification_results[i]['pred_label']
        refined_result_list.append('{} {} {} {} {}\n'.format(refined_class_name, x_center, y_center, w, h))

    with open(refined_result, "w") as refined_file:
        for line in refined_result_list:
           refined_file.write(line)