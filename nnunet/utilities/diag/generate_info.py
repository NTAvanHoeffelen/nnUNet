import json
import numpy as np
import glob
import pandas as pd
import math

def generate_csv(args, training_time):

    report_loc = args.prediction

    # TODO change this dir
    all_test_results_loc = '/mnt/netcache/bodyct/experiments/few_shot_segmentation_datasets/baseline_datasets'

    # get task name
    split_fileloc = report_loc.split('/')
    task_name = split_fileloc[-3]           # e.g. Task102_COPDGene_LeftSuperiorLobe_256_50_slices_1_pseudo_0

    dataset_json_loc = args.data + "/nnUNet_raw_data/" + task_name

    # Opening JSON file
    summary_file = open(report_loc + '/summary.json')
    dataset_file = open(dataset_json_loc + "/dataset.json")
    class_json_file = open(args.class_json)
    
    # load the JSON file
    sum_data = json.load(summary_file)
    dataset_data = json.load(dataset_file)
    class_json_data = json.load(class_json_file)

    # keep track of the Dice per scan
    dice_per_scan = []
    dice_per_pos_scan  = []
    fpr_per_scan = []
    #neg_total_positives = []
    for scan in sum_data['results']['all']:

        # check for nan
        if not math.isnan(scan['1']['Dice']):
            
            dice_per_scan.append(scan['1']['Dice'])

            # check if task was from a positive scan
            split_fileloc = scan['reference'].split('/')
            scan_name = split_fileloc[-1]
            if scan_name[-10:-7] != 'neg':
                dice_per_pos_scan.append(scan['1']['Dice'])
        
        if not math.isnan(scan['1']['False Positive Rate']):
            fpr_per_scan.append(scan['1']['False Positive Rate'])
         
        # split_fileloc = scan['reference'].split('/')
        # scan_name = split_fileloc[-1]
        # if scan_name[-10:-7] == 'neg':
        #     neg_total_positives.append(scan['1']['Total Positives Test'])

    # more information for the csv
    class_name = dataset_data['labels']['1']
    nr_slices = dataset_data['numTraining']
    folds = args.fold
    network_trainer = args.trainer
    positive_training_cases = class_json_data['train_slice_options']
    negative_training_cases = class_json_data['train_slice_options_negatives']
    
    positive_test_cases = class_json_data['test_slice_amount']
    negative_test_cases = class_json_data['test_slice_negative_amount']

    mean_testing_dice = format(sum_data['results']['mean']['1']['Dice'], '.4f')
    std_testing_dice = format(np.std(dice_per_scan), '.4f')

    mean_pos_testing_dice = format(np.mean(dice_per_pos_scan), '.4f')
    std_pos_testing_dice = format(np.std(dice_per_pos_scan), '.4f')

    mean_fpr = format(np.mean(fpr_per_scan), '.4f')
    std_fpr = format(np.std(fpr_per_scan), '.4f')

    summary_file.close()
    dataset_file.close()

    csv_destination_loc = report_loc
    csv_name = "/TESTING_RESULTS.csv"

    csv_all_testing_results = all_test_results_loc
    csv_all_data_name = "/ALL_TESTING_RESULTS.csv"

    df = pd.DataFrame({'Task Name': [task_name],
                   'Class': [class_name],
                   'Trainer': [network_trainer],
                   'Train Slices (pos/neg)': [f'{positive_training_cases} / {negative_training_cases}'],
                   'Test Slices (pos/neg)': [f'{positive_test_cases} / {negative_test_cases}'],
                   'Fold': [folds],
                   'Mean ± Std Dice (Test)': [f'{mean_testing_dice} ± {std_testing_dice}'],
                   'Mean ± Std Dice (only pos cases)': [f'{mean_pos_testing_dice} ± {std_pos_testing_dice}'],
                   'False Positive Rate': [f'{mean_fpr} ± {std_fpr}'],
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