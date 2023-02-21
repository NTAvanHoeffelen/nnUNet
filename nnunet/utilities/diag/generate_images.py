import glob
import numpy as np
from medpy import io
import shutil
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


plt.rcParams["figure.figsize"] = (40,16)
plt.rcParams.update({'font.size': 24})

# from (https://github.com/IML-DKFZ/nnunet-workshop/blob/main/nnU-Net_Workshop.ipynb)
def make_if_dont_exist(folder_path, overwrite=False, ):
        """
        creates a folder if it does not exists
        input: 
        folder_path : relative path of the folder which needs to be created
        over_write :(default: False) if True overwrite the existing folder (deleting all its contents!)
        """
        if os.path.exists(folder_path):

            if not overwrite:
                print(f"{folder_path} exists.")
            else:
                print(f"{folder_path} overwritten")
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
        else:
            os.makedirs(folder_path)
            print(f"{folder_path} created!")

def generate_svgs(args):
    ground_truth_dir = args.ground_truth
    prediction_dir = args.prediction
    image_dir = args.input

    destination_dir = args.prediction + "/images"

    make_if_dont_exist(destination_dir)

    list_of_ground_truths = sorted(glob.glob(ground_truth_dir + "/*.nii.gz"))
    list_of_predictions = sorted(glob.glob(prediction_dir + "/*.nii.gz"))
    list_of_images = sorted(glob.glob(image_dir + "/*.nii.gz"))

    for i in range(0, len(ground_truth_dir)):
        # load slice
        grount_truth_slice, _ = io.load(list_of_ground_truths[i])
        prediction_slice, _ = io.load(list_of_predictions[i])
        image_slice, _ = io.load(list_of_images[i])

        # find the middle slice in the image
        if image_slice.shape[2] == 1:
            selected_image_slice = image_slice[:,:,0]
        else:
            selected_image_slice = image_slice[:,:,int(image_slice.shape[2]//2)]


        # save svg of slice and annotation
        save_svg_slice_and_annotation(destination_dir, np.array(list_of_predictions)[i], selected_image_slice, grount_truth_slice[:,:,0], prediction_slice[:,:,0])


def save_svg_slice_and_annotation(svg_dest, fileloc, image_slice, ground_truth_annotation, predicted_annotation):

    split_fileloc = fileloc.split('\\')
    name = split_fileloc[-1]

    ground_truth_annotation_masked = np.ma.masked_where(ground_truth_annotation == 0, ground_truth_annotation)

    predicted_annotation_masked = np.ma.masked_where(predicted_annotation == 0, predicted_annotation)

    p = plt.get_cmap('hsv')

    gt_values = np.unique(ground_truth_annotation)
    pred_values = np.unique(predicted_annotation)

    plt.suptitle("{}".format(name[:-7]))
    plt.subplot(2,3,1)
    plt.imshow(image_slice, cmap='gray', vmin=-600-800, vmax=-600+800); plt.title("Slice")

    plt.subplot(2,3,2)
    plt.imshow(image_slice, cmap='gray', vmin=-600-800, vmax=-600+800)
    plt.subplot(2,3,2)
    plt.imshow(ground_truth_annotation_masked, cmap = p, alpha= 0.4); plt.title("Slice with Ground Truth annotation")

    plt.subplot(3,3,3)
    im = plt.imshow(ground_truth_annotation); plt.title("Ground Truth Annotation")

    # (https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend)
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in gt_values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Class {l}".format(l=gt_values[i]) ) for i in range(len(gt_values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


    plt.subplot(2,3,5)
    plt.imshow(image_slice, cmap='gray', vmin=-600-800, vmax=-600+800)
    plt.subplot(2,3,5)
    plt.imshow(predicted_annotation_masked, cmap = p, alpha= 0.4); plt.title("Slice with Predicted annotation")

    plt.subplot(2,3,6)
    im = plt.imshow(predicted_annotation); plt.title("Predicted Annotation")

    #(https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend)
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in pred_values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Class {l}".format(l=pred_values[i]) ) for i in range(len(pred_values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.savefig(svg_dest+ f"/{name[:-7]}.svg")

    