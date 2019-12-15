import os
import glob
import logging
import numpy as np
from time import time, strftime

import h5py
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tabulate import tabulate
from helpers import SVMHelper
from extractors import ExtractorsHelper
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels


class TrainingHelper:
    def __init__(self, gird_search=False, svm_model=False, knn_model=False, train_docs_h5='output/h5_features_data.h5', train_classes_h5='output/h5_labels_data.h5'):
        self.GirdSearchCV = gird_search
        self.SVMModel = svm_model
        self.kNNModel = knn_model
        self.train_docs = train_docs_h5
        self.train_classes = train_classes_h5
        self.svmhp = SVMHelper(kernel_type='linear')
        self.clf = SVC(C=0.01, kernel='linear', gamma=0.0001)
        self.run_controller()


    def stop_watch(self, start_time):
        ctime = time()
        duration = ctime - start_time
        if duration < 60:
            return 'Done {0:.2f} sec(s)'.format(duration)
        elif 60 < duration < 3600:
            duration = duration/60
            return 'Done {0:.2f} min(s)'.format(duration)
        else:
            duration = duration/(60*60)
            return 'Done {0:.2f} hr(s)'.format(duration)

    
    def data_reader(self, features_path, labels_path):
        if os.path.isfile(features_path) and os.path.isfile(labels_path):
            features_reader = h5py.File(features_path, 'r')
            labels_reader = h5py.File(labels_path, 'r')

            features_data = features_reader['features_vector']
            labels_data = labels_reader['labels']

            features_data = list(features_data)
            labels_data = list(labels_data)

            logging.info('Reading the features and successful labels.')
            return features_data, labels_data
        else:
            print('File not found.')

    
    def statistics_label(self, labels):
        headers = ['Ten loai benh', 'Nhan', 'So luong nhan/lop']
        l0 = 0; l1 = 0; l2 = 0; l3 = 0; l4 = 0
        for label in labels:
            if label == 0: l0 += 1
            elif label == 1: l1 += 1
            elif label == 2: l2 += 1
            elif label == 3: l3 += 1
            else: l4 += 1
        
        elements_label = len(np.unique(labels))
        total_label = l0 + l1 + l2 + l3 + l4
        table = [['Ghe nham', '0', l0],
                ['La khoe', '1', l1],
                ['Ray phan trang', '2', l2],
                ['Vang la gan xanh', '3', l3],
                ['Vang la thoi re', '4', l4],
                ['Tong cong:', elements_label, total_label]]
        logging.info('\n{}'.format(tabulate(table, headers, tablefmt='psql')))
        print(tabulate(table, headers, tablefmt='psql'), end='\n')


    def gird_search_cv(self):
        st = time()
        train_data, train_labels = self.data_reader(self.train_docs, self.train_classes)
        self.svmhp.train(train_data[:569], train_labels[:569])
        pred = self.svmhp.test(train_data[569:], train_labels[569:])
        logging.info('Gird search accuracy:{}'.format(pred))
        logging.info(self.stop_watch(st))


    def do_train_SVM_model(self):
        st = time()

        # Getting the dataset from h5 file
        train_data, train_label = self.data_reader(self.train_docs, self.train_classes)

        # Training model  
        logging.info('Fitting the SVM classifier to the training set')
        self.clf.fit(train_data[:569], train_label[:569])
        logging.info(self.clf)
        logging.info(self.stop_watch(st))

        # Dump model
        model = open('model/leaf_disease_detection.model', 'wb')
        joblib.dump(self.clf, model)
        model.close()

        # Predict the labels of testing dataset
        pred = self.clf.predict(train_data[569:])
        logging.info('Right labels:\n{}'.format(train_label[569:]))
        logging.info('Predict lables:\n{}'.format(pred))
        logging.info('Accuracy:{}'.format(accuracy_score(train_label[569:], pred)))
        logging.info('{}/{}'.format(accuracy_score(train_label[569:], pred, normalize=False), len(train_label[569:])))

        # Statistics all the labels of model
        logging.info('Statistics training labels:')
        self.statistics_label(train_label[:569])
        logging.info('Statistics testing labels:')
        self.statistics_label(train_label[569:])
        logging.info('Statistics predict labels')
        self.statistics_label(pred)
        logging.info('Confusion matrix:\n{}'.format(confusion_matrix(train_label[569:], pred)))


    def do_train_KNN_model(self):
        st = time()
        train_data, train_label = self.data_reader(self.train_docs, self.train_classes)

        k_range = list(range(1, 21))
        param_gird = dict(n_neighbors=k_range)

        kNN = GridSearchCV(KNeighborsClassifier(weights='distance'), param_gird, cv=2, verbose=True)

        # Training model
        logging.info('Fitting the KNN classifier to the training set')
        kNN = kNN.fit(train_data[:569], train_label[:569])
        logging.info(self.stop_watch(st))
        logging.info('Best estimator found by grid search:')
        logging.info(kNN.best_estimator_)

        # Predict the labels of testing dataset
        pred = kNN.best_estimator_.score(train_data[569:], train_label[569:])
        logging.info('Accuracy:{}'.format(pred))


    def run_controller(self):
        if self.SVMModel == True:
            self.do_train_SVM_model()
        elif self.kNNModel == True:
            self.do_train_KNN_model()
        elif self.GirdSearchCV == True:
            self.gird_search_cv()
        else:
            print('Choose a method to run.')
