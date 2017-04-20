import matplotlib.pyplot as plt
from scipy.misc import imresize

from src.metrics import get_area
import sys
import numpy as np


def get_training_images_from_intensities_set(normalized_set):
    images = []
    for i in range(len(normalized_set)):
        leave_one_out = normalized_set[np.arange(len(normalized_set)) != i]
        average = np.average(leave_one_out,0)
        image = []
        image.append(normalized_set[i,:])
        image.append(average)


        images.append(image)
    return np.array(images)


def get_training_set(dataset, image_width):
    training_images = []
    for sequence_str, sequence in dataset.items():
        for file_str, file in sequence.items():
            for charge_str, charge in file.items():
                if 'time' in charge_str:
                    continue
                intensities_set = []
                for ion_str, ion in charge.items():
                    intensities_set.append(ion['peak_intensities'])
                normalized_set = get_normalized_and_resized(intensities_set,image_width)
                for image in get_training_images_from_intensities_set(normalized_set):
                    training_images.append(image)

    return np.array(training_images)


def get_normalized_and_resized(intensities_set, size):
    intensities_set = np.array(intensities_set)

    max= np.max(intensities_set) + 20
    intensities_set /= max




    intensities_set *= 256
    result = []
    intensities_set = np.array(intensities_set, dtype='uint8')
    for intensities in intensities_set:
        resize = np.array(imresize(np.reshape(intensities, (1,len(intensities))), (1,size))[0],dtype=float)
        resize /= 256
        result.append(resize)

    return np.array(result)


