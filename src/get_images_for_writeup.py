import pickle
import numpy as np
from matplotlib import pyplot as plt

from src.training_set_generator import get_training_set

with open('data/smooth.pkl','rb') as smooth_file:
    smooth_dataset = pickle.load(smooth_file)

training_set = get_training_set(smooth_dataset,100)

t = np.arange(100)

image_one = training_set[2,:,:]
image_one_with_noise = image_one + np.random.normal(size = (2,100)) * 0.05



plt.plot(t,image_one[0,:])
plt.show()

plt.plot(t,image_one_with_noise[0,:], label="Transition")
plt.plot(t, image_one_with_noise[1,:], label="Average")

plt.legend()

plt.show()
