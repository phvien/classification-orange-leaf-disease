import logging
from time import time
from glob import glob
from random import randint
from os import path, listdir, makedirs

import h5py 
import joblib
import helpers
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


class ExtractorsHelper:

    def __init__(self):
        self.le = LabelEncoder()
        self.helpers_obj = helpers.FeaturesHelper()

    
    def stopwatch(self, start_time):
        current_time = time()
        duration = current_time - start_time
        
        if duration < 60:
            return 'Done {0:.2f}secs'.format(duration)
        elif 60 < duration < 3600:
            duration = duration/60
            return 'Done {0:.2f}mins'.format(duration)
        else:
            duration = duration/(60*60)
            return 'Done {0:.2f}hrs'.format(duration)
    

    def label_encoder(self, classes):
        self.le.fit(classes)
        return list(self.le.transform(classes))
    

    def shuffle(self, paths, classes):
        docs, docs_classes = shuffle(paths, classes, random_state=randint(1, 10))
        return docs, docs_classes
    

    def get_training_data(self, paths):
        training_paths = []
        training_classes = []
        classes = listdir(paths)
        logging.info('Receiving train data paths ...')
        for clname in classes:
            idx = 0
            subfolder = path.join(paths, clname)
            if path.exists(subfolder):
                print('\n[STATUS]-[Working in]:{}'.format(subfolder))
                for n_image, image in enumerate(listdir(subfolder)):
                    total_images = len(listdir(subfolder))
                    print('[{}][{}][{}]: {}'.format((n_image+1), total_images, clname, image))
                    impath = subfolder + '/' + image
                    training_paths.append(impath)
                    training_classes.append(clname)
                    idx += 1
                logging.info('Total images in [{}]:[{}]'.format(subfolder, idx))

                # Shuffle arrays or sparse matrices in a consistent way
                training_paths, training_classes = self.shuffle(training_paths, training_classes)

                # Encode labels with value between 0 and n_classes-1.
                training_labels = self.label_encoder(training_classes)
            else: 
                raise ValueError('The path {} does not exists.'.format(subfolder))
        return training_paths, training_labels


    def get_testing_data(self, paths):
        impaths = []
        if path.exists(paths):
            files = listdir(paths)
            for imfile in files:
                impath = path.join(paths, imfile)
                impaths.append(impath)
        else: 
            raise ValueError('The path must be string or exists.')
        logging.info(impaths)
        print(impaths)
        return impaths


    def writer_features(self, features, labels):
        if path.exists('output') is False:
            makedirs('output')
            
        fpath = 'output/h5_features_data.h5'
        lpath = 'output/h5_labels_data.h5'

        # Writing the features data
        fwriter = h5py.File(fpath, 'w')
        fwriter.create_dataset('features_vector', data=np.array(features))

        # Writing the labels data
        lwriter = h5py.File(lpath, 'w')
        lwriter.create_dataset('labels', data=np.array(labels))

        fwriter.close()
        lwriter.close()
        logging.info('The data writing successful.')
    

    def reader_features(self):
        fpath = 'output/h5_features_data.h5'
        lpath = 'output/h5_labels_data.h5'

        # Reading the features data
        freader = h5py.File(fpath, 'r')
        lreader = h5py.File(lpath, 'r')

        getf = freader['features_vector']
        getl = lreader['labels']

        freader.close()
        lreader.close()

        logging.info('The data reading successful.')
        return list(getf), list(getl)


    def label_statiscal(self, labels):
        l0 = 0; l1 = 0; l2 = 0; l3 = 0
        for i in labels:
            if i == 0:
                l0 += 1
            elif i == 1:
                l1 += 1
            elif i == 2:
                l2 += 1
            else:
                l3 += 1
        logging.info('Label statiscal per class:')
        logging.info('0-[{}], 1-[{}], 2-[{}], 3-[{}]'.format(l0, l1, l2, l3))
    

    def do_training_features_extraction(self, paths):
        training_paths = []
        training_classes = []
        start_time = time()
        training_paths, training_classes = self.get_training_data(paths)

        #--------------------------------------------------------------------
        logging.info('Training features extraction started ...')
        self.helpers_obj.develop_vocabulary(training_paths)
        training_features, training_labels = self.helpers_obj.build_training_features_classes_one_scaler(training_paths, training_classes)

        #--------------------------------------------------------------------
        if len(training_features) == 0 and len(training_labels) == 0:
            return [None, None]
        # Write data to .h5 file.
        self.writer_features(training_features, training_labels)
        logging.info('Shape of training_features {}'.format(np.array(training_features).shape))
        logging.info(self.stopwatch(start_time))

        #--------------------------------------------------------------------
        # Dump 'voc' variable
        f_codebook = open('output/code_book', 'wb')
        joblib.dump(self.helpers_obj.voc, f_codebook)
        f_codebook.close()

        # Dump 'scale' variable
        f_scale = open('output/scale', 'wb')
        joblib.dump(self.helpers_obj.scale, f_scale)
        f_scale.close()

    
    def do_testing_features_extraction(self, paths):
        start_time = time()
        impath = self.get_testing_data(paths)

        #--------------------------------------------------------------------
        # Load 'voc' variable created from do_training_features_extraction() method
        f_codebook = open('output/code_book', 'rb')
        code_book = joblib.load(f_codebook)
        self.helpers_obj = helpers.FeaturesHelper()
        self.helpers_obj.set_voc(code_book)

        # Load 'scale' variable
        f_scale = open('output/scale', 'rb')
        scale = joblib.load(f_scale)
        self.helpers_obj.set_scale(scale)

        #--------------------------------------------------------------------
        logging.info('Tesing features extraction started ...')
        testing_features = self.helpers_obj.build_testing_features(impath)
        logging.info('n_samples_seen_')
        logging.info(self.helpers_obj.scale.n_samples_seen_)

        #--------------------------------------------------------------------
        if len(testing_features) == 0:
            return [None, None]
        logging.info('Shape of testing_features: {}'.format(np.array(testing_features).shape))
        logging.info(self.stopwatch(start_time))

        return testing_features, impath
