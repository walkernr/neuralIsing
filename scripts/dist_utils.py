import numpy as np
from tensorflow.keras.utils import to_categorical


def sample_gaussian(num_rows, dimension):
    ''' unit gaussian sampling '''
    return np.random.normal(size=(num_rows, dimension))


def sample_categorical(num_rows, num_categories):
    ''' categorical sampling '''
    if num_categories > 0:
        sample = to_categorical(np.random.randint(0, num_categories, num_rows).reshape(-1, 1),
                                num_classes=num_categories)
    else:
        sample = np.empty(shape=(num_rows, num_categories))
    return sample


def sample_uniform(low, high, num_rows, dimension):
    ''' uniform sampling '''
    return np.random.uniform(low=low, high=high, size=(num_rows, dimension))