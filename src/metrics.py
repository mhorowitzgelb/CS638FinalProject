import numpy as np
import sys
from chromatogram_parser import get_dataset

def get_area(intensities):
    sum = 0
    for i in range(len(intensities) - 1):
        sum += (intensities[i] + intensities[i+1])/2.0
    return sum


def get_dot(areas_a, areas_b):
    squared_norm_a = 0
    squared_norm_b = 0
    dot = 0
    count = 0
    #print("Dot correlation")
    for ion_str in areas_a.keys():
        if not ion_str in areas_b:
            continue
        area_a = areas_a[ion_str]
        area_b = areas_b[ion_str]
        #print("Area a: {0} , Area b: {1}".format(area_a,area_b))
        dot += area_a * area_b
        squared_norm_a += area_a ** 2
        squared_norm_b += area_b ** 2
        count += 1
    if(count <= 1):
        return -1

    dot /= (np.sqrt(squared_norm_a)* np.sqrt(squared_norm_b))
    #print("Correlation {0}".format(dot))
    return dot

def get_dot_products(areas_set):
    count = 0
    sum = 0
    for i in range(len(areas_set)):
        for j in range(i + 1, len(areas_set)):
            areas_a = areas_set[i]
            areas_b = areas_set[j]
            dot = get_dot(areas_a,areas_b)

            if dot == -1:
                continue
            sum += dot
            count += 1
    if count == 0:
        print("Count for dot products was 0")
        return -1
    return sum /count

def get_average_dot_product(dataset):
    orig_count = 0
    pred_count = 0;
    sum_orig = 0.0
    sum_pred = 0.0
    for sequence_str in dataset.keys():
        sequence = dataset[sequence_str]
        charge_areas_orig = {}
        charge_areas_pred = {}
        for file_str in sequence.keys():
            file = sequence[file_str]
            for charge_str in file.keys():
                if 'time' in charge_str:
                    continue
                charge = file[charge_str]
                if not charge_str in charge_areas_orig:
                    charge_areas_orig[charge_str] = []
                if not charge_str in charge_areas_pred:
                    charge_areas_pred[charge_str] = []
                orig_areas = {}
                pred_areas = {}
                charge_areas_orig[charge_str].append(orig_areas)
                charge_areas_pred[charge_str].append(pred_areas)
                for ion_str in charge:
                    ion = charge[ion_str]
                    orig_areas[ion_str] = (get_area(ion['peak_intensities']))
                    pred_areas[ion_str] = get_area(ion['peak_intensities_pred'])

        for charge_str in charge_areas_orig.keys():
            areas_set_orig = charge_areas_orig[charge_str]
            if len(areas_set_orig) < 2:
                print("areas set to small")
                continue
            average_correlation = get_dot_products(areas_set_orig)
            if(average_correlation == -1):
                print("not adding average correlation")
                continue
            orig_count +=1
            sum_orig += average_correlation
        for charge_str in charge_areas_pred.keys():
            areas_set_pred = charge_areas_pred[charge_str]
            if len(areas_set_orig) < 2:
                print("areas set to small")
                continue
            average_correlation = get_dot_products(areas_set_pred)
            if(average_correlation == -1):
                print("not adding average correlation")
                continue
            pred_count +=1
            sum_pred += average_correlation
    return (sum_orig / orig_count, sum_pred / pred_count)
'''
import pickle

dataset = pickle.load(open('data/ManualHannesDataset.pkl','rb'))

print(get_average_dot_product(dataset))
'''
