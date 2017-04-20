import csv
import sys
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
time_step_size = 0.01


def get_peak_intensities(all_times, all_intensities, start, end):
    i = 0
    time = 0
    next = 0
    offset = 0;
    peak_intensities = []
    peak_times = []
    while all_times[i+1] < start:
        i +=1
    while next < end:
        time = all_times[i]
        if(i+1 >= len(all_times)):
            break;
        next = all_times[i +1]
        time_width = next - time
        j = 0
        while True:
            a = (offset + j * time_step_size)/ time_width
            if( a >= 1 or time + a*time_width > end):
                break
            peak_intensities.append((1-a) * all_intensities[i] + a * all_intensities[i+1])
            peak_times.append((1-a) * all_times[i] + a * all_times[i+1])
            j += 1
        offset = (a-1) * time_width
        i += 1

    return (peak_times,peak_intensities)



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
            if(row[2] == '#N/A'):
                continue
            start_time = float(row[2])
            end_time = float(row[3])
            if not sequence in dataset:
                dataset[sequence] = {}

            dataset[sequence][file_name]= {}
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
            charge = row[2]
            ion = row[4]
            all_times = row[8]
            all_times = list(map(float, all_times.split(",")))
            all_intensities = row[9]
            all_intensities = list(map(float,all_intensities.split(',')))
            if not sequence in dataset:
                continue
            if not file_name in dataset[sequence]:
                continue
            (peak_times, peak_intensities) = get_peak_intensities(all_times, all_intensities,
                                               dataset[sequence][file_name]["start_time"], dataset[sequence][file_name]["end_time"])
            if not charge in dataset[sequence][file_name]:
                dataset[sequence][file_name][charge] = {}
            dataset[sequence][file_name][charge][ion] = {}
            dataset[sequence][file_name][charge][ion]['peak_intensities'] = peak_intensities
            dataset[sequence][file_name][charge][ion]['peak_times'] = peak_times
    return dataset
'''
import random
import pickle
dataset = pickle.load(open('data/ManualHannesDataset.pkl','rb'))#get_dataset("data/smoothBoundaries.csv", 'data/smoothCalibration.tsv')

import pickle

#pickle.dump(dataset, open('data/smooth.pkl', 'wb'))

sequence = random.choice(list(dataset.keys()))

while(True):

    key = sequence
    intensities = dataset[key]
    #print(type(intensities))

    key = random.choice(list(intensities.keys()))
    intensities = intensities[key]
    #print(type(intensities))


    key = random.choice(list(intensities.keys()))
    while('time' in key):
        key = random.choice(list(intensities.keys()))
    intensities = intensities[key]

    #print(type(intensities))

    #print(intensities)

    print("Drawing")
    for a in intensities.keys():
        b = intensities[a]
        c = b['peak_intensities']
        times_actual = b["peak_times"]
        times = np.arange(len(c))
        print('{0} Start: {1} , End: {2}, Points: {3} '.format(a, times_actual[0], times_actual[-1], len(times_actual)))
        plt.plot(times,c)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
'''