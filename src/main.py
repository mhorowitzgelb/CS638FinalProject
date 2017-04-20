import pickle
import numpy as np
from src.training_set_generator import get_training_set
import matplotlib.pyplot as plt
import sys


def main(args):
    with open('data/smooth.pkl','rb') as smooth_file:
        dataset = pickle.load(smooth_file)


    desired_image_width = 100
    #Gets a training set of {training_set_size}x2x100
    training_set = get_training_set(dataset,desired_image_width)

    #number of training exmamples
    num_training_examples = training_set.shape[0]

    #Channels should be 2
    number_of_channels = training_set.shape[1]

    #Should be same as desired_image_width
    image_width = training_set.shape[2]


    example_one = training_set[0,:,:]

    #plot one unique transition
    #x is values 0 ... 99 , normalized time values
    #y is the transitions intensities at each time point
    plt.plot(np.arange(image_width), example_one[0,:], label='single transition')

    #Plot average shape over all other transitions of the peptide
    plt.plot(np.arange(image_width), example_one[1,:], label='shape average')

    plt.legend()

    plt.show(block=False)
    plt.waitforbuttonpress()
    sys.exit()

if __name__ == '__main__':
    main(None)