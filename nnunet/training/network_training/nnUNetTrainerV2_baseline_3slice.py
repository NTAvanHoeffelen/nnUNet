#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D, DataLoader2D_pseudo_3d_fix
from collections import OrderedDict
import numpy as np
from typing import Tuple
from nnunet.training.network_training.nnUNetTrainerV2_baseline import nnUNetTrainerV2_baseline

class nnUNetTrainerV2_baseline_3slice(nnUNetTrainerV2_baseline):
    def __init__(self,  plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
        self.pseudo_3d = True
        self.pseudo_3d_slices = 3 # e.g. 5 slices --> 2 above, 1 middle, 2 below
        self.avg_slices = False
        
        if self.pseudo_3d:
            if self.avg_slices == True:
                self.use_custom_mask_for_norm = OrderedDict([(i, False) for i in range(0, 3)])
            else:
                self.use_custom_mask_for_norm = OrderedDict([(i, False) for i in range(0, self.pseudo_3d_slices)])

        # when using 2d data (which we are) this should be true
        self.unpack_data = True
    def setup_DA_params(self):
        super().setup_DA_params()
        if self.pseudo_3d:
            self.data_aug_params["mask_was_used_for_normalization"] =  self.use_custom_mask_for_norm #self.use_mask_for_norm

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = self.new_batch_size
        #self.patch_size = self.new_patch_size

        if self.pseudo_3d:
            self.num_input_channels = self.pseudo_3d_slices
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            if self.pseudo_3d:
                dl_tr = DataLoader2D_pseudo_3d_fix(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,  pseudo_3d_slices = self.pseudo_3d_slices,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', avg_slices=self.avg_slices)
                dl_val = DataLoader2D_pseudo_3d_fix(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,  pseudo_3d_slices = self.pseudo_3d_slices,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', avg_slices=self.avg_slices)
            else:
                dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')


        return dl_tr, dl_val