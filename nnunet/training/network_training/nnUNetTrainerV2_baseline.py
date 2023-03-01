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

class nnUNetTrainerV2_baseline(nnUNetTrainerV2):
    def __init__(self,  plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        self.new_batch_size = 16
        #self.new_patch_size = np.array([512, 512]).astype(int)
        self.max_num_epochs = 25
        # self.pseudo_3d = False
        # self.pseudo_3d_slices = 1 # e.g. 5 slices --> 2 above, 1 middle, 2 below
        # self.avg_slices = False
        
        # if self.pseudo_3d:
        #     if self.avg_slices == True:
        #         self.use_custom_mask_for_norm = OrderedDict([(i, False) for i in range(0, 3)])
        #     else:
        #         self.use_custom_mask_for_norm = OrderedDict([(i, False) for i in range(0, self.pseudo_3d_slices)])

        # when using 2d data (which we are) this should be true
        self.unpack_data = True


    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False  # DISABLE MIRRORING
        # if self.pseudo_3d:
        #     self.data_aug_params["mask_was_used_for_normalization"] =  self.use_custom_mask_for_norm #self.use_mask_for_norm

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = self.new_batch_size
        #self.patch_size = self.new_patch_size
        # if self.pseudo_3d:
        #   self.num_input_channels = self.pseudo_3d_slices
    
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        do_mirroring = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    # def get_basic_generators(self):
    #     self.load_dataset()
    #     self.do_split()

    #     if self.threeD:
    #         dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
    #                              False, oversample_foreground_percent=self.oversample_foreground_percent,
    #                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
    #         dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
    #                               oversample_foreground_percent=self.oversample_foreground_percent,
    #                               pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
    #     else:
    #         if self.pseudo_3d:
    #             dl_tr = DataLoader2D_pseudo_3d_fix(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,  pseudo_3d_slices = self.pseudo_3d_slices,
    #                                 oversample_foreground_percent=self.oversample_foreground_percent,
    #                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', avg_slices=self.avg_slices)
    #             dl_val = DataLoader2D_pseudo_3d_fix(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,  pseudo_3d_slices = self.pseudo_3d_slices,
    #                                 oversample_foreground_percent=self.oversample_foreground_percent,
    #                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', avg_slices=self.avg_slices)
    #         else:
    #             dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
    #                                 oversample_foreground_percent=self.oversample_foreground_percent,
    #                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
    #             dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
    #                                 oversample_foreground_percent=self.oversample_foreground_percent,
    #                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')


    #     return dl_tr, dl_val
    
    