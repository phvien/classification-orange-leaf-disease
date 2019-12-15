import numpy as np
import os
import glob
import logging
from random import shuffle
from time import time, strftime

import h5py
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle


logging.basicConfig(filename='log_libsvm_data_format.log', level=logging.DEBUG)
logging.debug('='*100)
logging.debug('[START TIME]: {}' .format(strftime('%H:%M:%S %d-%m-%Y')))

START_TIME = time()
TRAIN_DATA_DIR = 'dataset/train'
TEST_DATA_DIR = 'dataset/test'

FEATURES_PATH = 'output/training_features.h5'
LABELS_PATH = 'output/training_labels.h5'


# Import the feature vector and trained labels
h5_features  = h5py.File(FEATURES_PATH, 'r')
h5_labels = h5py.File(LABELS_PATH, 'r')

get_features = h5_features['Features_vector']
get_labels = h5_labels['Labels']

features = np.array(get_features)
labels = np.array(get_labels)

h5_features.close()
h5_labels.close()

print('Shape of features vector:', features.shape)
print('Shape of labels:', labels.shape)
print('Labels:', labels)


def randomizing(data, label):
    data_sparse = coo_matrix(data)
    x_data, x_sparse, y_label = shuffle(data, data_sparse, label, random_state=10)
    return x_data, y_label


def libsvm_data_format(feature_data, label_data, output_file):
    """Shape of feature_data: (num_rows, num_col_features)
    Shape of label_data: (num_rows,)"""

    logging.info('Start converting data format to libsvm format...')
    features, labels = randomizing(feature_data, label_data)
    
    if os.path.isfile(output_file):
        os.remove(output_file)
    
    f = open(output_file, 'a+')
    label_count = []
    all_data = []
    for n_row, label in enumerate(labels):
        new_line = []
        new_line.append(str(label))
        idx = 0
        for feature_point in features[n_row]:
            idx += 1
            if feature_point != 0.00:
                item = ('{}:{}' .format(idx, feature_point))
                new_line.append(item)
        new_line = ' '.join(new_line)
        new_line += '\n'
        f.write(new_line)
        all_data.append(new_line)
        label_count.append(label)
    print(label_count)
    f.close()
    logging.info('Number of labels received: {}' .format(len(label_count)))
    print('End of libsvm data format.')
    return all_data


def libsvm_format(output_file, feature_data, label_data=None):
    label_count = []
    all_data = []
    if label_data is None:
        pass
    else:
        logging.info('Start converting data format to libsvm format...')
        features, labels = randomizing(feature_data, label_data)
        if os.path.isfile(output_file):
            os.remove(output_file)
        
        f = open(output_file, 'a+')
        for n_row, label in enumerate(labels):
            new_line = []
            new_line.append(str(label))
            idx = 0
            for feature_point in features[n_row]:
                idx += 1
                if feature_point != 0.00:
                    item = ('{}:{}' .format(idx, feature_point))
                    new_line.append(item)
            new_line = ' '.join(new_line)
            new_line += '\n'
            f.write(new_line)
            all_data.append(new_line)
            label_count.append(label)
        print(label_count)
        f.close()
        logging.info('Number of labels received: {}' .format(len(label_count)))
        print('End of libsvm data format.')
    return all_data



libsvm_format('kasfsdjf', 'hkdsfjhksadf')


END_TIME = time()
dur = END_TIME - START_TIME
logging.debug('[END TIME]: {}' .format(strftime('%H:%M:%S %d-%m-%Y')))
if dur < 60:
    print('Execution Time: {0:.2f} secs' .format(dur))
    logging.debug('[EXECUTION TIME]: {0:.2f}secs' .format(dur))
elif 60 < dur < 3600:
    dur = dur/60
    print('Execution Time: {0:.2f} mins' .format(dur))
    logging.debug('[EXECUTION TIME]: {0:.2f}mins' .format(dur))
else:
    dur = dur/(60*60)
    logging.debug('[EXECUTION TIME]: {}hrs' .format(dur))
    print('Execution Time: {0:.2f} hrs' .format(dur))
    