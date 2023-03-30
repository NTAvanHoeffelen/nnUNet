import glob
import numpy as np
from medpy import io
import shutil
import os
import json

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import gc
import math

plt.rcParams["figure.figsize"] = (40,16)
plt.rcParams.update({'font.size': 24})


# from (https://github.com/IML-DKFZ/nnunet-workshop/blob/main/nnU-Net_Workshop.ipynb)
def make_if_dont_exist(folder_path, overwrite=False):
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

    # open and load summary file
    summary_file_loc = prediction_dir + '/summary.json'
    summary_file = open(summary_file_loc)
    summary_data = json.load(summary_file)

    make_if_dont_exist(destination_dir)

    list_of_ground_truths = sorted(glob.glob(ground_truth_dir + "/*.nii.gz"))
    list_of_predictions = sorted(glob.glob(prediction_dir + "/*.nii.gz"))
    list_of_images = sorted(glob.glob(image_dir + "/*.nii.gz"))

    if len(list_of_images) == len(list_of_ground_truths):
        mid_slice_id = str(0)
    else:
        mid_slice_id = str(int(len(list_of_images)//len(list_of_ground_truths)) - 2)

    while len(mid_slice_id) < 4:
        mid_slice_id = "0" + mid_slice_id
    
    list_of_images = sorted(glob.glob(image_dir + f"/*{mid_slice_id}.nii.gz"))

    vmin = args.window_min
    vmax = args.window_max

    # gerate a multipage pdf:
    with PdfPages(prediction_dir + '/All_test_results.pdf') as pdf:
        for i in range(0, len(list_of_ground_truths)):
            # load slice
            grount_truth_slice, _ = io.load(list_of_ground_truths[i])
            prediction_slice, _ = io.load(list_of_predictions[i])
            image_slice, _ = io.load(list_of_images[i])

            # get the dice of slice
            dice_score_slice, _, _ = get_info_fast(summary_data, i)

            # save svg of slice and annotation
            save_svg_slice_and_annotation(destination_dir, np.array(list_of_predictions)[i], image_slice[:,:,0], grount_truth_slice[:,:,0], prediction_slice[:,:,0], dice_score_slice, pdf, vmin, vmax)

            # free up memory (IMPORTANT)
            del grount_truth_slice
            del prediction_slice
            del image_slice
            gc.collect()

    # close summary file
    summary_file.close()
    
    plt.close()

def generate_summary_pdf(args):
    # filter for positve or negative cases

    ground_truth_dir = args.ground_truth
    prediction_dir = args.prediction
    image_dir = args.input

    destination_dir = args.prediction + "/images"

    # open and load summary file
    summary_file_loc = prediction_dir + '/summary.json'
    summary_file = open(summary_file_loc)
    summary_data = json.load(summary_file)

    make_if_dont_exist(destination_dir)

    list_of_ground_truths = sorted(glob.glob(ground_truth_dir + f"/*.nii.gz"))
    list_of_images = sorted(glob.glob(image_dir + "/*.nii.gz"))

    if len(list_of_images) == len(list_of_ground_truths):
        mid_slice_id = str(0)
    else:
        mid_slice_id = str(int(len(list_of_images)//len(list_of_ground_truths)) - 2)

    while len(mid_slice_id) < 4:
        mid_slice_id = "0" + mid_slice_id
    
    list_of_predictions = sorted(glob.glob(prediction_dir + f"/*.nii.gz"))
    list_of_ground_truths = sorted(glob.glob(ground_truth_dir + f"/*.nii.gz"))
    list_of_images = sorted(glob.glob(image_dir + f"/*{mid_slice_id}.nii.gz"))

    vmin = args.window_min
    vmax = args.window_max

    # Find the index of each positive and negative case
    scan_counter = 0
    pos_scan_indices = []
    neg_scan_indices = []
    for scan in summary_data['results']['all']:
        split_fileloc = scan['reference'].split('/')
        scan_name = split_fileloc[-1]
        if scan_name[-10:-7] == 'pos':
            pos_scan_indices.append(scan_counter)
        else:
            neg_scan_indices.append(scan_counter)
        scan_counter += 1

    dice_pos_scans = []
    fnr_pos_scans = []
    fpr_pos_scans = []
    non_nan_pos_scan_indices = []
    # get dice of each positive scan (but not the nan dice values)
    for indx in pos_scan_indices:
        # check for nan
        if not math.isnan(summary_data['results']['all'][indx]['1']['Dice']):
            
            dice_pos_scans.append(summary_data['results']['all'][indx]['1']['Dice'])
            fnr_pos_scans.append(summary_data['results']['all'][indx]['1']['False Negative Rate'])
            fpr_pos_scans.append(summary_data['results']['all'][indx]['1']['False Positive Rate'])
            non_nan_pos_scan_indices.append(indx)

    # sort indices from smallest to biggest dice
    dice_indices = np.argsort(dice_pos_scans)

    # flip indices from biggest to smallest dice
    dice_indices = np.flip(dice_indices)

    # calculate the indices when picking nr_pics
    nr_pics = 5
    step = len(dice_indices)//(nr_pics -1)
    pic_indices = [0]
    for i in range(1, nr_pics):
        pic_indices.append(int(step*i) - 1)

    with PdfPages(prediction_dir + '/Summary_pos_test_cases.pdf') as pdf:
        for i in pic_indices:

            indx = non_nan_pos_scan_indices[dice_indices[i]]

            grount_truth_slice, _ = io.load(list_of_ground_truths[indx])
            prediction_slice, _ = io.load(list_of_predictions[indx])
            image_slice, _ = io.load(list_of_images[indx])

            # get the dice of slice
            dice_score_slice = dice_pos_scans[dice_indices[i]]
            fnr_slice = fnr_pos_scans[dice_indices[i]]
            fpr_slice = fpr_pos_scans[dice_indices[i]]

            # save svg of slice and annotation
            save_svg_slice_and_annotation_summary(np.array(list_of_predictions)[indx],
                                                  image_slice[:,:,0],
                                                  grount_truth_slice[:,:,0],
                                                  prediction_slice[:,:,0],
                                                  dice_score_slice,
                                                  fnr_slice,
                                                  fpr_slice,
                                                  pdf,
                                                  i,
                                                  len(list_of_ground_truths),
                                                  vmin,
                                                  vmax)
            
            # free up memory (IMPORTANT)
            del grount_truth_slice
            del prediction_slice
            del image_slice
            gc.collect()
    
    neg_fnr = []
    neg_fpr = []

    # get dice of each positive scan (but not the nan dice values)
    for indx in neg_scan_indices:
        # check for nan
        neg_fnr.append(summary_data['results']['all'][indx]['1']['False Negative Rate'])
        neg_fpr.append(summary_data['results']['all'][indx]['1']['False Positive Rate'])

     # sort indices from smallest to biggest dice
    sorted_neg_fpr = np.argsort(neg_fpr)

    # flip indices from biggest to smallest dice
    sorted_neg_fpr = np.flip(sorted_neg_fpr)

    # calculate the indices when picking nr_pics
    nr_pics = 5
    step = len(sorted_neg_fpr)//(nr_pics -1)
    pic_indices = [0]
    for i in range(1, nr_pics):
        pic_indices.append(int(step*i) - 1)

    with PdfPages(prediction_dir + '/Summary_neg_test_cases.pdf') as pdf:
        for i in pic_indices:

            indx = neg_scan_indices[sorted_neg_fpr[i]]

            grount_truth_slice, _ = io.load(list_of_ground_truths[indx])
            prediction_slice, _ = io.load(list_of_predictions[indx])
            image_slice, _ = io.load(list_of_images[indx])

            slice_neg_fpr = neg_fpr[sorted_neg_fpr[i]]
            slice_neg_fnr = neg_fnr[sorted_neg_fpr[i]]

            # save svg of slice and annotation
            save_svg_slice_and_annotation_summary(np.array(list_of_predictions)[indx],
                                                  image_slice[:,:,0],
                                                  grount_truth_slice[:,:,0],
                                                  prediction_slice[:,:,0],
                                                  0,
                                                  slice_neg_fnr,
                                                  slice_neg_fpr,
                                                  pdf,
                                                  i,
                                                  len(list_of_ground_truths),
                                                  vmin,
                                                  vmax)
            
            # free up memory (IMPORTANT)
            del grount_truth_slice
            del prediction_slice
            del image_slice
            gc.collect()

    # close summary file
    summary_file.close()
    
    plt.close()

def get_dice(summary_data, gt_reference_loc):

    split_gt_reference = gt_reference_loc.split('/')
    gt_reference_name = split_gt_reference[-1]

    for slice in summary_data['results']['all']:
        slice_reference = slice['reference']
        split_slice_reference = slice_reference.split('/')
        slice_reference_name = split_slice_reference[-1]

        if gt_reference_name == slice_reference_name:
            return slice['1']['Dice']
    return 0

def get_info_fast(summary_data, i):
    # works on the assumption the items in the summary and in the directory are sorted in the same manner.
    DICE = summary_data['results']['all'][i]['1']['Dice']
    FNR = summary_data['results']['all'][i]['1']['False Negative Rate']
    FPR = summary_data['results']['all'][i]['1']['False Positive Rate']
    return DICE, FNR, FPR 

def save_svg_slice_and_annotation(svg_dest, fileloc, image_slice, ground_truth_annotation, predicted_annotation, dice_score_slice, pdf, vmin, vmax):

    split_fileloc = fileloc.split('/')
    name = split_fileloc[-1]

    ground_truth_annotation_masked = np.ma.masked_where(ground_truth_annotation == 0, ground_truth_annotation)

    predicted_annotation_masked = np.ma.masked_where(predicted_annotation == 0, predicted_annotation)

    p = plt.get_cmap('hsv')
    p2 = plt.get_cmap('brg')

    gt_values = np.unique(ground_truth_annotation)
    pred_values = np.unique(predicted_annotation)

    fig = plt.figure()

    plt.suptitle("{}; Dice {:.4f}; Level: {} -- {}".format(name[:-7], dice_score_slice, vmin, vmax))
    plt.subplot(2,3,1)
    plt.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax); plt.title("Slice")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(2,3,2)
    plt.imshow(ground_truth_annotation_masked, cmap = p, alpha= 0.4); plt.title("Slice with Ground Truth annotation")
    plt.axis('off')

    plt.subplot(2,3,3)
    im = plt.imshow(ground_truth_annotation); plt.title("Ground Truth Annotation")
    plt.axis('off')

    # (https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend)
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in gt_values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Class {l}".format(l=gt_values[i]) ) for i in range(len(gt_values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


    plt.subplot(2,3,4)
    plt.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(2,3,4)
    plt.imshow(ground_truth_annotation_masked, cmap = p, alpha= 0.4)
    plt.subplot(2,3,4)
    plt.imshow(predicted_annotation_masked, cmap = p2, alpha= 0.45); plt.title("Slice with both annotations")
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(2,3,5)
    plt.imshow(predicted_annotation_masked, cmap = p2, alpha= 0.45); plt.title("Slice with Predicted annotation")
    plt.axis('off')

    plt.subplot(2,3,6)
    im = plt.imshow(predicted_annotation); plt.title("Predicted Annotation")
    plt.axis('off')

    #(https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend)
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in pred_values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Class {l}".format(l=pred_values[i]) ) for i in range(len(pred_values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.savefig(svg_dest+ f"/{name[:-7]}.png")

    # add to pdf
    pdf.savefig(fig)

    plt.clf()
    plt.cla()
    plt.close()

    del ground_truth_annotation_masked
    del predicted_annotation_masked
    del p
    del p2
    del pred_values
    del gt_values
    del im
    del colors
    del patches
    del image_slice
    del ground_truth_annotation
    del predicted_annotation
    del fig
    gc.collect()

# retrieved from: https://stackoverflow.com/questions/16267143/matplotlib-single-colored-colormap-with-saturation on 07-03-2023
def generate_custom_cmap(from_rgb,to_rgb):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def save_svg_slice_and_annotation_summary(fileloc, image_slice, ground_truth_annotation, predicted_annotation, dice_score_slice, fnr_slice, fpr_slice, pdf, dice_nr, total_test_size, vmin, vmax):

    split_fileloc = fileloc.split('/')
    name = split_fileloc[-1]

    # set gt to 2
    ground_truth_annotation[ground_truth_annotation == 1] = 2

    # combine the masks    
    combined_annotation = ground_truth_annotation + predicted_annotation

    # extract tp, fp, fn
    tp = np.ma.masked_where(combined_annotation != 3, combined_annotation) # yellow
    fn = np.ma.masked_where(combined_annotation != 2, combined_annotation) # green
    fp = np.ma.masked_where(combined_annotation != 1, combined_annotation) # red
    
    p = generate_custom_cmap([1, 1, 0], [1, 1, 0])
    p2 = generate_custom_cmap([0, 1, 0], [0, 1, 0])
    p3 =  generate_custom_cmap([1, 0, 0], [1, 0, 0])

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.suptitle("{}; #{}/{} Dice {:.4f}; FNR {:.4f}; FPR {:.4f}; Level: {} -- {}".format(name[:-7], dice_nr+1, total_test_size, dice_score_slice, fnr_slice, fpr_slice, vmin, vmax))
    ax.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax); ax.set_title("Slice")
    ax.axis('off')

    ax2 = fig.add_subplot(122)
    ax2.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    im_tp = ax2.imshow(tp, cmap = p, alpha= 0.5, label= 'TP')
    im_fn = ax2.imshow(fn, cmap = p2, alpha= 0.5, label= 'FN')
    im_fp = ax2.imshow(fp, cmap = p3, alpha= 0.5, label= 'FP'); ax.set_title("Slice with both annotations")
    ax2.axis('off')

    #(https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend)
    # get the colors of the values, according to the 
    # colormap used by imshow
    vals = [1,1,1]
    labels = ["TP", "FN", "FP"]
    colors = [im_tp.cmap(im_tp.norm(1)), im_fn.cmap(im_tp.norm(1)), im_fp.cmap(im_tp.norm(1))]
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{}".format(labels[i]) ) for i in range(len(vals)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    # add to pdf
    pdf.savefig(fig)

    plt.clf()
    plt.cla()
    plt.close()

    del p
    del p2
    del p3
    del tp
    del fp
    del fn
    del image_slice
    del ground_truth_annotation
    del predicted_annotation
    del fig
    gc.collect()