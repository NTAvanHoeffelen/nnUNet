import json
import numpy as np
import glob
import pandas as pd
import math

def generate_csv(args, training_time):

    report_loc = args.prediction

    # TODO change this dir
    all_test_results_loc = '/mnt/netcache/bodyct/experiments/few_shot_segmentation_datasets/full_datasets/COPDGene/Classes'

    # get task name
    split_fileloc = report_loc.split('/')
    task_name = split_fileloc[-3]           # e.g. Task102_COPDGene_LeftSuperiorLobe_256_50_slices_1_pseudo_0

    dataset_json_loc = args.data + "/nnUNet_raw_data/" + task_name

    # Opening JSON file
    summary_file = open(report_loc + '/summary.json')
    dataset_file = open(dataset_json_loc + "/dataset.json")
    
    # load the JSON file
    sum_data = json.load(summary_file)
    dataset_data = json.load(dataset_file)

    # keep track of the Dice per scan
    dice_per_scan = []
    dice_per_pos_scan  = []
    neg_total_positives = []
    for scan in sum_data['results']['all']:

        # check for nan
        if not math.isnan(scan['1']['Dice']):
            
            dice_per_scan.append(scan['1']['Dice'])

            # check if task was from a positive scan
            split_fileloc = scan['reference'].split('/')
            scan_name = split_fileloc[-1]
            if scan_name[-10:-7] != 'neg':
                dice_per_pos_scan.append(scan['1']['Dice'])
         
        split_fileloc = scan['reference'].split('/')
        scan_name = split_fileloc[-1]
        if scan_name[-10:-7] == 'neg':
            neg_total_positives.append(scan['1']['Total Positives Test'])

    # more information for the csv
    class_name = dataset_data['labels']['1']
    nr_slices = dataset_data['numTraining']
    folds = args.fold
    network_trainer = args.trainer
    mean_testing_dice = sum_data['results']['mean']['1']['Dice']
    mean_pos_testing_dice = np.mean(dice_per_pos_scan)
    mean_total_pos_per_neg_scan = np.mean(neg_total_positives)
    std_testing_dice = np.std(dice_per_scan)
    std_pos_testing_dice = np.std(dice_per_pos_scan)

    summary_file.close()
    dataset_file.close()

    csv_destination_loc = report_loc
    csv_name = "/TESTING_RESULTS.csv"

    csv_all_testing_results = all_test_results_loc
    csv_all_data_name = "/ALL_TESTING_RESULTS.csv"

    df = pd.DataFrame({'Task Name': [task_name],
                   'Class': [class_name],
                   'Trainer': [network_trainer],
                   'Slices': [nr_slices],
                   'Fold': [folds],
                   'Mean Dice (Test)': [mean_testing_dice],
                   'Mean Dice (only pos cases)': [mean_pos_testing_dice],
                   'STD Dice (Test)': [std_testing_dice],
                   'STD Dice (only pos cases)': [std_pos_testing_dice],
                   'Mean positive pixels in negative cases': [int("{:.4f}".format(mean_total_pos_per_neg_scan))],
                   'Training time (in seconds)': [training_time],
                   'NOTE': [args.note]
                   })
    df.to_csv(csv_destination_loc + csv_name ,index=False)

    try:
        all_testing_results_df = pd.read_csv(csv_all_testing_results + csv_all_data_name, sep = ",", on_bad_lines='skip')
        frames = [all_testing_results_df, df]
        all_testing_results_df = pd.concat(frames, axis = 0)

        all_testing_results_df.to_csv(csv_all_testing_results + csv_all_data_name, index = False)
    except:
        df.to_csv(csv_all_testing_results + csv_all_data_name, index = False)