from torch.utils.data import Sampler, WeightedRandomSampler
import torch
import numpy as np
from mmengine.dist import get_dist_info, sync_random_seed
import math

from typing import Sequence, Iterator, Sized, Optional

from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class MalariaWeightedRandomSampler(Sampler):

    def __init__(self,
                 dataset: Sized,
                 seed: Optional[int] = None,
                 round_up: bool = True,) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)


        # print(len(self.dataset))

        # print("DATA FROM DATASET:", self.dataset[0]['data_samples'].gt_label)
        
        self.calculate_weight()

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        self.sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=self.num_samples,
            replacement=True,  # Allows oversampling
            generator = g
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        # if self.shuffle:
        #     g = torch.Generator()
        #     g.manual_seed(self.seed + self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # else:
        #     indices = torch.arange(len(self.dataset)).tolist()

        #Just sample with weights for each class
        indices = list(self.sampler)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def calculate_weight(self):
        labels = []
        for i in range(len(self.dataset)):
            labels.append(self.dataset[i]['data_samples'].gt_label.item())
            # if i % 10000 == 0:
            #     print(i)
        labels = np.array(labels)
        
        class_counts = np.bincount(labels)
        self.class_weights = 1.0 / class_counts
        # Modify weight of last two classes (other and difficult)
        # July 22, 2025: Incorrectly modify class weight? 
        # self.class_weights[:-2] = self.class_weights[:-2] / 20
        self.class_weights[5:] = self.class_weights[5:] / 20
        
        # Which values for their weights are suitable? 

        # labels = np.array([self.dataset[i]['data_samples'].gt_label.item() for i in range(len(self.dataset))])
        #print(labels)
        self.sample_weights = self.class_weights[labels]
        
        # print("Class count:", class_counts)
        # print("Class weights:", self.class_weights)
        # print("Sample Weights", self.sample_weights[:10])