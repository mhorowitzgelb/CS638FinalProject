import matplotlib.pyplot as plt
from scipy.misc import imresize

from metrics import get_area
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

    return training_images


def do_prediction(image, sess, generator_model):
    return generator_model.predict(image,sess)
    #return np.random.rand(100)

total_sequences_predicted_for = 1000

def do_predictions(dataset, sess, model, image_width):
    i = 0
    for sequence_str, sequence in dataset.items():
        i += 1
        if i > total_sequences_predicted_for:
            break
        else:
            dataset[sequence_str]["predicted"] = True

        for file_str, file in sequence.items():
            if(file_str is "predicted"):
                continue
            for charge_str, charge in file.items():
                if 'time' in charge_str:
                    continue
                intensities_set = []
                ion_strings = []
                for ion_str, ion in charge.items():
                    intensities_set.append(ion['peak_intensities'])
                    ion_strings.append(ion_str)
                normalized_set = get_normalized_and_resized(intensities_set,image_width)
                for idx, image in enumerate(get_training_images_from_intensities_set(normalized_set)):
                    pred = do_prediction(image, sess,model)
                    dataset[sequence_str][file_str][charge_str][ion_strings[idx]]['peak_intensities_pred'] = pred



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

import pickle
from metrics import  get_average_dot_product
import matplotlib.pyplot as plt
if __name__ == '__main__':
    import pickle

    with open("testPred.pkl", 'rb') as predFile:
        predictions = pickle.load(predFile)

    for sequence_str, sequence in predictions.items():
        for file_str, file in sequence.items():
            if (file_str is "predicted"):
                continue
            for charge_str, charge in file.items():
                if 'time' in charge_str:
                    continue
                intensities_set = []
                ion_strings = []
                for ion_str, ion in charge.items():
                    intensities_set.append(ion['peak_intensities'])
                    ion_strings.append(ion_str)
                normalized_set = get_normalized_and_resized(intensities_set, 100)
                for idx, image in enumerate(get_training_images_from_intensities_set(normalized_set)):
                    pred = predictions[sequence_str][file_str][charge_str][ion_strings[idx]]['peak_intensities_pred']
                    input = image[0,:]
                    t= np.arange(100)
                    f, axarr = plt.subplots(2)
                    axarr[0].plot(t, input)
                    axarr[0].set_title('Input')
                    axarr[1].plot(t, pred)
                    axarr[1].set_title('Output')
                    plt.show()

