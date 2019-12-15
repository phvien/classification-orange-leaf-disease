import logging
from os import path
from time import time

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from extractors import ExtractorsHelper


class PredictHelper:

    def __init__(self, paths, classes, model_path):
        self.paths = paths
        self.classes = classes
        self.model_path = model_path
        self.ehelper = ExtractorsHelper()
        self.clf = None
        self.y_pred = None
        self.predict_helper()

    
    def load_figure(self, impaths, titles):
        plt.figure()
        num_cols = len(impaths)
        for i in range(0, len(impaths)):
            plt.subplot(1, num_cols, i+1)
            im = mpimg.imread(impaths[i])
            plt.title(titles[i])
            plt.imshow(im)
        # Show plot in fullscreen
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()

    
    def show_figure(self, paths, labels):
        impaths = []
        titles = []
        for i in range(0, len(paths)):
            impaths.append(paths[i])
            titles.append(self.classes[labels[i]])
        self.load_figure(impaths, titles)


    def load_model(self):
        model = open(self.model_path, 'rb')
        self.clf = joblib.load(model)


    def predict_helper(self):
        start_time = time()
        data, impaths = self.ehelper.do_testing_features_extraction(self.paths)
        self.load_model()
        self.y_pred = self.clf.predict(data)
        print(self.ehelper.stopwatch(start_time))
        self.show_figure(impaths, self.y_pred)
