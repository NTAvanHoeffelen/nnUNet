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

class nnUNetTrainerV2_baseline_7slice(nnUNetTrainerV2_baseline):
    def __init__(self,  plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = self.new_batch_size
        #self.patch_size = self.new_patch_size
        # if self.pseudo_3d:
        self.num_input_channels = 7
    
    