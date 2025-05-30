from ultralytics import YOLO
import argparse
import os
import cv2


# Create argument parser
parser = argparse.ArgumentParser(description="Parser for classification")

# Add arguments
parser.add_argument("--detection_model", type=str, help="detection model")
parser.add_argument("--blood_smear_images", type=str, help="path to folder with blood smear images")

args = parser.parse_args()

detection_model = YOLO(args.detection_model)

detection_results = detection_model.predict(source=args.blood_smear_images, save= True, save_txt= True)

save_dir = detection_results[0].save_dir
txt_result_dir = os.path.join(save_dir, "labels")
txt_file_list = [f for f in os.listdir(txt_result_dir) if os.path.isfile(os.path.join(txt_result_dir, f))]
print("Number of blood smear images:", len(txt_file_list))

for txt_file in txt_file_list:
    img_name = [f for f in os.listdir(args.blood_smear_images) if f.startswith(txt_file.split('.')[0])][0]
    img_path = os.path.join(args.blood_smear_images, img_name)
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    output_folder = os.path.join(save_dir, 'crop', txt_file.split('.')[0])
    os.makedirs(output_folder, exist_ok=True)
    print("Save cropped RBCs to:", output_folder)

    result_file = os.path.join(txt_result_dir, txt_file)
    with open(result_file, "r") as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()

        class_name, x_center, y_center, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x_center, y_center, w, h = int(x_center * width), int(y_center * height), int(w * width), int(h * height)
        
        # Get top-left and bottom-right coordinates
        x1, y1 = max(0, x_center - w // 2), max(0, y_center - h // 2)
        x2, y2 = min(width, x_center + w // 2), min(height, y_center + h // 2)
        
        # Crop object
        cropped_object = image[y1:y2, x1:x2]

        output_filename = os.path.join(output_folder, f"{class_name}_{i}.jpg")
        cv2.imwrite(output_filename, cropped_object)