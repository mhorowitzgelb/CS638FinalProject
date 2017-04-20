from src.metrics import get_area
import numpy as np
def get_training_set(dataset, image_width):
    peak_lengths = []
    for sequence_str, sequence in dataset.items():
        for file_str, file in sequence.items():
            for charge_str, charge in file.items():
                if 'time' in charge_str:
                    continue
                intensities = []
                for ion_str, ion in charge.items():
                    intensities.append(ion['peak_intensities'])
                    peak_lengths.append(len(intensities))
                    break;

                #intensities = np.array(intensities)
    return np.array(intensities)
