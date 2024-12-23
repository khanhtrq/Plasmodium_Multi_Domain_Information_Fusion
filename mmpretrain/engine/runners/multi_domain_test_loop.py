from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop, TestLoop
from mmengine.logging import HistoryBuffer
from mmengine.hooks import Hook
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.evaluator import Evaluator

from torch.utils.data import DataLoader
import torch
import glob
import cv2
import numpy as np

from mmengine.runner.amp import autocast
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of

from ...models.classifiers.multi_domain_classifier import MultiDomainClassifier


import os

import shutil

CLASS_NAMES = ['Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 
               'HealthyRBC', 'Other', 'Difficult']
DOMAIN_NAMES = ['OurPlasmodium', 'BBBC041', 'IMLMalaria']

@LOOPS.register_module()
class MultiDomainTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 dataloaders_multi_domain: List[Union[DataLoader, Dict]] = [],
                 domain_names: List = None, #List of domain names according to dataloaders
                 blood_smear_data_path: str = None, 
                 save_false_cases_name: bool = False,
                 fp16: bool = False) -> None:
        
        super().__init__(runner, dataloader, evaluator)

        self.n_domains = len(dataloaders_multi_domain)
        self.dataloaders = [] #List of all dataloaders

        #build dataloaders
        for dataloader in dataloaders_multi_domain:
            if isinstance(dataloader, dict):
                self.dataloaders.append(runner.build_dataloader(dataloader, seed=runner.seed))
            else:
                self.dataloaders.append(dataloader)
        
        if domain_names is None: 
            self.domain_names = DOMAIN_NAMES
        else:
            self.domain_names = domain_names

        self.save_false_cases_name = save_false_cases_name
        self.blood_smear_data_path = blood_smear_data_path

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()

        #Khanh implementation for multi-domain evaluation 
        metrics_all = {}
        for domain_idx, dataloader in enumerate(self.dataloaders):
            for idx, data_batch in enumerate(dataloader):
                self.run_iter(idx, data_batch, domain_idx)

                # '''
                # ----------------------
                # Break for testing only
                # ----------------------
                # '''
                # break

            # compute metrics
            metrics = self.evaluator.evaluate(len(dataloader.dataset))

            for metric_name in metrics.keys():
                metrics_all['{}/{}'.format(self.domain_names[domain_idx], metric_name)] = metrics[metric_name]

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)
        
        #Copying false classification cases
        metrics_all = self.save_false_classification(metrics_all)
                                    
        self.runner.call_hook('after_test_epoch', metrics=metrics_all)
        self.runner.call_hook('after_test')
        return metrics
    
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict],
                 domain_idx: int) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement

        with autocast(enabled=self.fp16):
            if isinstance(self.runner.model, MultiDomainClassifier):
                outputs = self.runner.model.test_step(data_batch, domain_idx = domain_idx)
            else: 
                outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)



    def save_false_classification(self, metrics_all):
        save_dir_cell = create_folder_with_suffix(os.path.join(self.runner.work_dir, 'false_classification'))
        save_dir_blood_smear = create_folder_with_suffix(os.path.join(self.runner.work_dir, 'false_classification_blood_smear'))
    
        for domain_idx in range(len(self.dataloaders)):
            metric_name = '{}/{}'.format(self.domain_names[domain_idx], 'false_classification')
            
            if metric_name in metrics_all.keys():
                for gt in metrics_all[metric_name]:
                    for pred in metrics_all[metric_name][gt]:
                        path_cell = os.path.join(save_dir_cell, 
                                            self.domain_names[domain_idx], CLASS_NAMES[gt], 
                                            CLASS_NAMES[pred])
                        path_blood_smear = os.path.join(save_dir_blood_smear, 
                                            self.domain_names[domain_idx], CLASS_NAMES[gt], 
                                            CLASS_NAMES[pred])
                        if len(metrics_all[metric_name][gt][pred]) > 0:
                            os.makedirs(path_cell, exist_ok=True)
                            os.makedirs(path_blood_smear, exist_ok=True)

                        for i in range(len(metrics_all[metric_name][gt][pred])):  
                            if (gt != pred): 
                                img_path = metrics_all[metric_name][gt][pred][i]
                                cell_img = cv2.imread(img_path)

                                shutil.copy(img_path, os.path.join(path_cell, '{}.jpg'.format(i))) 
                                                            
                                #In case running on Window
                                img_path = img_path.replace("\\", '/')
                                # print("IMAGE PATH AFTER BEING REPLACED:", img_path)
                                img_path = img_path.split('/')

                                if (self.domain_names[domain_idx] == 'OurPlasmodium') and (self.blood_smear_data_path is not None):
                                    blood_img_path = os.path.join(self.blood_smear_data_path, 
                                                                img_path[-4], 'images', img_path[-3] + '.*')
                                    blood_img_name = img_path[-3]
                                    blood_img_path = glob.glob(blood_img_path)[0]

                                    # print(img_path)
                                    # print(blood_img_path)
                                    blood_img = cv2.imread(blood_img_path)
                                    text = 'Label: {}, Predicted: {}, Image: {}'.format(CLASS_NAMES[gt], 
                                                                            CLASS_NAMES[pred],
                                                                            img_path[-3])

                                    img = concat_images(blood_img, cell_img, text= text)
                                    counter = 1
                                    saved_path = os.path.join(path_blood_smear, '{}_{}.jpg'.format(blood_img_name, counter))
                                    while os.path.exists(saved_path):
                                        counter += 1
                                        saved_path = os.path.join(path_blood_smear, '{}_{}.jpg'.format(blood_img_name, counter))

                                    cv2.imwrite(saved_path, img)                           

                                elif self.domain_names[domain_idx] == 'BBBC041':
                                    pass
                                elif self.domain_names[domain_idx] == 'IMLMalaria':
                                    pass

                            # if (gt == pred) and (gt == 4) and (i == 30):
                            #     break 
                        if gt == pred:
                            metrics_all[metric_name][gt][pred] = []

                # If original cropped cells are not needed
                if not self.save_false_cases_name:
                    metrics_all.pop(metric_name)
        return metrics_all

def _update_losses(outputs: list, losses: dict) -> Tuple[list, dict]:
    """Update and record the losses of the network.

    Args:
        outputs (list): The outputs of the network.
        losses (dict): The losses of the network.

    Returns:
        list: The updated outputs of the network.
        dict: The updated losses of the network.
    """
    if isinstance(outputs[-1],
                  BaseDataElement) and outputs[-1].keys() == ['loss']:
        loss = outputs[-1].loss  # type: ignore
        outputs = outputs[:-1]
    else:
        loss = dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value:
                losses[loss_name].update(loss_value_i.item())
    return outputs, losses

def _parse_losses(losses: Dict[str, HistoryBuffer],
                  stage: str) -> Dict[str, float]:
    """Parses the raw losses of the network.

    Args:
        losses (dict): raw losses of the network.
        stage (str): The stage of loss, e.g., 'val' or 'test'.

    Returns:
        dict[str, float]: The key is the loss name, and the value is the
        average loss.
    """
    all_loss = 0
    loss_dict: Dict[str, float] = dict()

    for loss_name, loss_value in losses.items():
        avg_loss = loss_value.mean()
        loss_dict[loss_name] = avg_loss
        if 'loss' in loss_name:
            all_loss += avg_loss

    loss_dict[f'{stage}_loss'] = all_loss
    return loss_dict

def create_folder_with_suffix(base_folder_name):
    folder_name = base_folder_name
    counter = 1

    # Keep checking if the folder exists
    while os.path.exists(folder_name):
        folder_name = f"{base_folder_name}_{counter}"  # Append a suffix
        counter += 1

    # Create the new folder
    os.makedirs(folder_name)
    return folder_name

def concat_images(img1, img2, text=None, space=50, axis=1, font_scale=1, color=(0, 0, 0), thickness=2, max_width=400):
    """
    Concatenate two images with optional white space between them.
    
    Parameters:
    - img1: First image (numpy array).
    - img2: Second image (numpy array).
    - space: White space between images (default is 50 pixels).
    - axis: The axis along which to concatenate (1 for horizontal, 0 for vertical).
    
    Returns:
    - concatenated image (numpy array).
    """

    if text:
        h2, w2, c2 = img2.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Wrap text using the external wrap_text function
        wrapped_text, max_width = wrap_text(text, font, font_scale, thickness, max_width)
        
        # Calculate the height required for the text
        text_height = sum(cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in wrapped_text)
        text_height += len(wrapped_text) * 5  # Add spacing between lines

        # Adjust canvas dimensions dynamically
        img2_width = max(w2, max_width)
        img2_height = h2 + text_height + 20
        canvas_img2 = np.ones((img2_height, img2_width, 3), dtype=np.uint8) * 255

        # Place img2 on the canvas
        canvas_img2[0:h2, 0:w2] = img2

        # Add the wrapped text
        y_offset = h2 + 20
        for line in wrapped_text:
            cv2.putText(canvas_img2, line, (10, y_offset), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            (_, line_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            y_offset += line_height + 5

        # Update img2 with the new canvas
        img2 = canvas_img2


    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    w2 = int(h1/4*w2/h2)
    h2 = int(h1/4)

    img2 = cv2.resize(img2, (w2, h2))
    
    # If concatenating horizontally (axis=1), the heights should be the same
    if axis == 1:
        canvas_height = max(h1, h2)
        canvas_width = w1 + w2 + space
        # Create a white canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place the first image on the left
        canvas[0:h1, 0:w1] = img1
        
        # Place the second image on the right with space in between
        canvas[0:h2, w1 + space:w1 + space + w2] = img2
    
    # If concatenating vertically (axis=0), the widths should be the same
    elif axis == 0:
        canvas_width = max(w1, w2)
        canvas_height = h1 + h2 + space
        # Create a white canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place the first image on the top
        canvas[0:h1, 0:w1] = img1
        
        # Place the second image on the bottom with space in between
        canvas[h1 + space:h1 + space + h2, 0:w2] = img2
    
    return canvas


def wrap_text(text, font, font_scale, thickness, max_width):
    """
    Wrap text into multiple lines based on a maximum width.
    Dynamically adjusts max_width if a single word exceeds it.

    Parameters:
    - text: The input text string to wrap.
    - font: Font type for `cv2.putText`.
    - font_scale: Scale of the font.
    - thickness: Thickness of the font.
    - max_width: Maximum width for wrapping text.

    Returns:
    - lines: List of wrapped lines of text.
    - updated_max_width: The updated maximum width to fit the longest word.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    updated_max_width = max_width

    for word in words:
        # Get size of the word
        (word_width, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
        if word_width > updated_max_width:
            # Update max_width dynamically if a word is too long
            updated_max_width = word_width + 20  # Add padding for the longest word

        # Test line width with the new word
        test_line = current_line + " " + word if current_line else word
        (test_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if test_width <= updated_max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines, updated_max_width
