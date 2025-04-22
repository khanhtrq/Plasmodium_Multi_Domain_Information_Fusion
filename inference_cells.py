from mmpretrain.apis import ImageClassificationInferencer
import argparse
from pathlib import Path
import os


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--cls_model", type=str, help="path to config file")
parser.add_argument("--cls_pretrained", type=str, help="path to cls_model cls_pretrained (trained parameteres) pt file")
parser.add_argument("--rbc_images", type=str, help="path to image or images folder")
parser.add_argument("--detection_save_dir", type=str, help="path to image or images folder")


args = parser.parse_args()

inferencer = ImageClassificationInferencer(
    model = args.cls_model,
    pretrained = args.cls_pretrained,
    device='cuda')
txt_result_dir = os.path.join(args.detection_save_dir, "labels")


for rbc_folder in os.listdir(args.rbc_images):
    if Path(args.rbc_images).is_file():
        input_images = [args.rbc_images]
    else:
        folder_path = os.path.join(args.rbc_images, rbc_folder)
        input_images = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                input_images.append(os.path.abspath(os.path.join(root, file)))


    classification_results = inferencer(inputs = input_images,
                                        show_dir = './visualize/')
        #----------
    # DETECTION RSULT
    # ---------
    
    txt_file = [f for f in os.listdir(os.path.join(args.detection_save_dir, "labels")) if f.startswith(rbc_folder)][0]
    result_file = os.path.join(txt_result_dir, txt_file)
    
    os.makedirs(os.path.join(args.detection_save_dir, "life_cycle_labels"), exist_ok=True)
    refined_result = os.path.join(os.path.join(args.detection_save_dir, "life_cycle_labels"), txt_file)
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