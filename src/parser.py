import csv
import numpy as np

def get_dataset(peak_range_file, intensities_file):
    dataset = {}
    #read in times
    with open(peak_range_file, newline='')  as range_file:
        reader = csv.reader(range_file, delimiter=',')
        read_header = False
        for row in reader:
            if not read_header:
                read_header = True
                continue
            file_name = row[0]
            sequence = row[1]
            start_time = float(row[2])
            end_time = float(row[3])
            if not sequence in dataset:
                dataset[sequence] = {}
            dataset[sequence][file_name] = {}
            dataset[sequence][file_name]["start_time"] = start_time
            dataset[sequence][file_name]["end_time"] = end_time


    with open(intensities_file, newline='') as intensities_file:
        reader = csv.reader(intensities_file, delimiter='\t')
        read_header = False
        for row in reader:
            if not read_header:
                read_header = True
                continue
            file_name = row[0]
            sequence = row[1]
            all_times = row[8]
            all_times = list(map(float, all_times.split(",")))
            all_intensities = row[9]
            all_intensities = list(map(float,all_intensities.split(',')))

            intensities = get_peak_intensities(all_times, all_intensities,
                                               dataset[sequence][file_name]["start_time"], dataset[sequence][file_name]["end_time"])

            print(all_times)
            counter += 1
            if counter >= 1:
                break


get_dataset("../data/ManualHannes.csv",'../data/ManualHannes.tsv')