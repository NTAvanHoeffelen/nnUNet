import json
import numpy as np
import glob
import pandas as pd

def generate_csv(args, training_time):

    report_loc = args.prediction

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
    for scan in sum_data['results']['all']:
        dice_per_scan.append(scan['1']['Dice'])


    # more information for the csv
    class_name = dataset_data['labels']['1']
    nr_slices = dataset_data['numTraining']
    folds = args.fold
    network_trainer = args.trainer
    mean_testing_dice = sum_data['results']['mean']['1']['Dice']
    std_testing_dice = np.std(dice_per_scan)

    summary_file.close()
    dataset_file.close()

    csv_destination_loc = report_loc
    csv_name = "/TESTING_RESULTS.csv"

    df = pd.DataFrame({'Task Name': [task_name],
                   'Class': [class_name],
                   'Trainer': [network_trainer],
                   'Slices': [nr_slices],
                   'Fold': [folds],
                   'Mean Dice (Test)': [mean_testing_dice],
                   'STD Dice (Test)': [std_testing_dice],
                   'Training time (in seconds)': [training_time]
                   })
    df.to_csv(csv_destination_loc + csv_name ,index=False)

