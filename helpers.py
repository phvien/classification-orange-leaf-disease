import cv2
from numpy import *
from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from time import time
import os.path
import imghdr
from skimage import feature
from skimage import exposure
from PIL import Image
import leargist
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input as pre_in_resnet
from keras.applications.inception_v3 import InceptionV3, preprocess_input as pre_in_inception
from keras.applications.xception import Xception, preprocess_input as pre_in_xception  # TensorFlow ONLY
from keras.applications.vgg16 import VGG16, preprocess_input as pre_in_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as pre_in_vgg19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import makedirs
import joblib
import logging

class ImageHelper:
    def __init__(self, width, model_name="resnet"):
        self.img_width = width
        self.sift_object = cv2.xfeatures2d.SIFT_create()
        # define a dictionary that maps model names to their classes
        # inside Keras
        self.MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception,  # TensorFlow ONLY
            "resnet": ResNet50
        }
        # initialize the input image shape (224x224 pixels) along with
        # the pre-processing function (this might need to be changed
        # based on which model we use to classify our image)
        self.inputShape = (224, 224)
        self.preprocess = imagenet_utils.preprocess_input
        if model_name == "vgg16":
            self.preprocess = pre_in_vgg16
        if model_name == "vgg19":
            self.preprocess = pre_in_vgg19
        if model_name == "inception":
            self.preprocess = pre_in_inception
            self.inputShape = (299, 299)
        if model_name == "xception":
            self.preprocess = pre_in_xception
            self.inputShape = (299, 299)
        if model_name == "resnet":
            self.preprocess = pre_in_resnet
        # esnure a valid model name was supplied via command line argument
        if model_name not in self.MODELS.keys():
            raise AssertionError("The model_name argument should "
                                 "be a key in the `MODELS` dictionary")
        # load our the network weights from disk (NOTE: if this is the
        # first time you are running this script for a given network, the
        # weights will need to be downloaded first -- depending on which
        # network you are using, the weights can be 90-575MB, so be
        # patient; the weights will be cached and subsequent runs of this
        # script will be *much* faster)
        self.Network = self.MODELS[model_name]
        self.model = self.Network(weights="imagenet", include_top=False)

    def is_valid_image(self, path):
        if os.path.isfile(path):
            t = imghdr.what(path)
            if t is not None:
                if t != 'gif':
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def scale(self, image, width):
        if image.shape[1] > width:
            r = float(width) / float(image.shape[1])
            dim = (width, int(r * image.shape[0]))
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        else:
            return image

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def color_feature(self, fn):
        img = cv2.imread(fn)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        return hist

    def sift_feature(self, fn):
        img = cv2.imread(fn)
        si = self.scale(img, self.img_width)
        gi = self.gray(si)
        return self.sift_object.detectAndCompute(gi, None)

    def hog_feature(self, fn, width, height):
        img = cv2.imread(fn)
        dim = (width, height)
        si = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        gi = self.gray(si)
        return feature.hog(gi, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(1, 1), block_norm='L2-Hys')

    def visualize_hog_feature(self, fn):
        img = cv2.imread(fn)

        (hf, hi) = feature.hog(img, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=True, multichannel=True)
        hogImage = exposure.rescale_intensity(hi, in_range=(0, 10))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 15))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    def gist_feature(self, fn):
        im = Image.open(fn)
        return leargist.color_gist(im)

    def deep_feature(self, fn):
        # load the input image using the Keras helper utility while ensuring
        # the image is resized to `inputShape`, the required input dimensions
        # for the ImageNet pre-trained network
        image = load_img(fn, target_size=self.inputShape)
        image = img_to_array(image)
        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through thenetwork
        image = expand_dims(image, axis=0)
        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        image = self.preprocess(image)
        # now load the output of the NEXT-TO-TOP layer
        outputs = self.model.predict(image)
        features = outputs.flatten()
        return features

class FeaturesHelper:
    def __init__(self, img_width=300, n_clusters=400, model_name='resnet', sift=True, color=True, hog=True, gist=True,
                 deep=True):
        self.img_width = img_width
        self.n_clusters = n_clusters
        self.model_name = model_name
        self.sift = sift
        self.color = color
        self.hog = hog
        self.gist = gist
        self.deep = deep
        self.imghp = ImageHelper(self.img_width, self.model_name)
        self.rng = random.RandomState(0)
        self.kmeans_obj = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.rng, verbose=True)
        self.scale = None
        self.voc = None
        self.idf = None
        self.training_paths = []
        self.training_features = []
        self.training_classes = []
        self.testing_paths = []
        self.testing_features = []
        self.scaler = StandardScaler()

    def develop_vocabulary(self, paths):
        logging.info('Learning the SIFT BOVW dictionary... ')
        t0 = time()
        if self.voc is None:
            logging.info(self.kmeans_obj)
            buff = []
            n_points = 0
            i = 0
            for path in paths:
                # logging.info('Loading vocabulary:' + path)
                i = i + 1
                if self.imghp.is_valid_image(path):
                    kp, desc = self.imghp.sift_feature(path)
                    if len(kp) > 0:
                        n_points += len(kp)
                        buff.append(desc)
                if (i % 100 == 0 and i > 0) or (i == len(paths) - 1):
                    descriptors = array(buff[0])  # stack all features for k-means
                    for j in arange(1, len(buff)):
                        descriptors = vstack((descriptors, buff[j]))
                    self.kmeans_obj.partial_fit(descriptors)
                    buff = []
                    n_points = 0
                    logging.info('Partial fit of %4i out of %i' % (i, len(paths)))

        dt = time() - t0
        logging.info('Done in %.2fs.' % dt)
        self.voc = self.kmeans_obj.cluster_centers_
        logging.info('Vocabulary Histogram Generated')

    def set_voc(self, voc):
        self.voc = voc

    def project(self, descriptors):
        """ Project descriptors on the vocabulary
            to create a histogram of words. """

        # histogram of image words
        imhist = zeros((self.n_clusters))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1
        return imhist

    def get_words(self, descriptors):
        """ Convert descriptors to words. """
        return vq(descriptors, self.voc)[0]

    def build_training_features_classes(self, paths, classes):
        self.training_paths = []
        self.training_features = []
        self.training_classes = []

        logging.info('Extracting features ...')

        if (self.sift) and (self.voc is None):
            logging.info('Done! No vocabulary to build SIFT BOVW features!')
            return [None, None]
        else:
            sift_imwords = []
            other_imwords = []
            t0 = time()
            index = 0
            n_images = 0
            for i in range(0, len(paths)):
                path = paths[i]
                # logging.info('Loading ' + path)
                if self.imghp.is_valid_image(path):
                    ok = True
                    sift_features = []
                    other_features = []
                    if self.sift:
                        kp, desc = self.imghp.sift_feature(path)
                        if len(kp) > 0:
                            n_images += 1
                            data = desc
                            sift_features = self.project(data)
                        else:
                            ok = False
                    if self.color:
                        colorf = self.imghp.color_feature(path)
                        if len(colorf > 0):
                            other_features = concatenate((other_features, colorf), axis=None)
                        else:
                            ok = False
                    if self.hog:
                        hogf = self.imghp.hog_feature(path, 80, 80)
                        if len(hogf > 0):
                            other_features = concatenate((other_features, hogf), axis=None)
                        else:
                            ok = False
                    if self.gist:
                        gistf = self.imghp.gist_feature(path)
                        if len(gistf > 0):
                            other_features = concatenate((other_features, gistf), axis=None)
                        else:
                            ok = False
                    if self.deep:
                        deepf = self.imghp.deep_feature(path)
                        if len(deepf > 0):
                            other_features = concatenate((other_features, deepf), axis=None)
                        else:
                            ok = False
                    if ok:
                        if self.sift:
                            sift_imwords.append(sift_features)
                        if self.color or self.hog or self.gist or self.deep:
                            other_imwords.append(other_features)
                        self.training_classes.append(classes[i])
                        self.training_paths.append(paths[i])
                        print(self.training_paths[i], self.training_classes[i])
                if index % 100 == 0:
                    logging.info('Getting image features of %4i out of %i' % (index, len(paths)))
                index += 1
            dt = time() - t0
            all_sift_features = []
            all_other_features = []
            if len(sift_imwords) > 0:
                nparray = array(self.standardize(sift_imwords))
                nbr_occurences = sum((nparray > 0) * 1, axis=0)
                self.idf = log((1.0 * n_images) / (1.0 * nbr_occurences + 1))
                nparray = nparray * self.idf
                all_sift_features = nparray
            if len(other_imwords) > 0:
                all_other_features = array(self.standardize(other_imwords))
            if len(all_sift_features) > 0 and len(all_other_features) > 0:
                self.training_features = hstack([all_sift_features, all_other_features])
            else:
                if len(all_sift_features) > 0 and len(all_other_features) == 0:
                    self.training_features = all_sift_features
                if len(all_sift_features) == 0 and len(all_other_features) > 0:
                    self.training_features = all_other_features
                if len(all_sift_features) == 0 and len(all_other_features) == 0:
                    self.training_features = []
            logging.info('Done in %.2fs.' % dt)
        return self.training_features, self.training_classes

    def build_training_features_classes_one_scaler(self, paths, classes):
        self.training_paths = []
        self.training_features = []
        self.training_classes = []

        logging.info('Extracting features ...')

        if (self.sift) and (self.voc is None):
            logging.info('Done! No vocabulary to build SIFT BOVW features!')
            return [None, None]
        else:
            imwords = []
            t0 = time()
            for i in range(0, len(paths)):
                path = paths[i]
                # logging.info('Loading ' + path)
                if self.imghp.is_valid_image(path):
                    ok = True
                    features = []
                    if self.sift:
                        kp, desc = self.imghp.sift_feature(path)
                        if len(kp) > 0:
                            data = desc
                            features = concatenate((features, self.project(data)), axis=None)
                        else:
                            ok = False
                    if self.color:
                        colorf = self.imghp.color_feature(path)
                        if len(colorf > 0):
                            features = concatenate((features, colorf), axis=None)
                        else:
                            ok = False
                    if self.hog:
                        hogf = self.imghp.hog_feature(path, 80, 80)
                        if len(hogf > 0):
                            features = concatenate((features, hogf), axis=None)
                        else:
                            ok = False
                    if self.gist:
                        gistf = self.imghp.gist_feature(path)
                        if len(gistf > 0):
                            features = concatenate((features, gistf), axis=None)
                        else:
                            ok = False
                    if self.deep:
                        deepf = self.imghp.deep_feature(path)
                        if len(deepf > 0):
                            features = concatenate((features, deepf), axis=None)
                        else:
                            ok = False
                    if ok:
                        imwords.append(features)
                        self.training_classes.append(classes[i])
                        self.training_paths.append(paths[i])
                        print('[{}][{}][{}][{}]:{}'.format((i+1), len(paths), classes[i], len(features), paths[i]))
                        logging.info('[{}][{}][{}][{}]:{}'.format((i+1), len(paths), classes[i], len(features), paths[i]))
                if i % 100 == 0 and i > 0:
                    logging.info('Getting image features of %4i out of %i' % (i, len(paths)))
            if len(imwords) > 0:
                self.training_features = array(self.standardize(imwords))
            else:
                self.training_features = []
            dt = time() - t0
            logging.info('Done in %.2fs.' % dt)
        return self.training_features, self.training_classes

    def build_testing_features(self, paths):
        self.testing_paths = []
        self.testing_features = []
        logging.info('Extracting features ...')

        if (self.sift) and (self.voc is None):
            logging.info('Done! No vocabulary to build SIFT BOVW features!')
            return [None, None]
        else:
            sift_imwords = []
            other_imwords = []
            t0 = time()
            index = 0
            n_images = 0
            for i in range(0, len(paths)):
                path = paths[i]
                # logging.info('Loading ' + path)
                if self.imghp.is_valid_image(path):
                    ok = True
                    sift_features = []
                    other_features = []
                    if self.sift:
                        kp, desc = self.imghp.sift_feature(path)
                        if len(kp) > 0:
                            n_images += 1
                            data = desc
                            sift_features = self.project(data)
                        else:
                            ok = False
                    if self.color:
                        colorf = self.imghp.color_feature(path)
                        if len(colorf > 0):
                            other_features = concatenate((other_features, colorf), axis=None)
                        else:
                            ok = False
                    if self.hog:
                        hogf = self.imghp.hog_feature(path, 80, 80)
                        if len(hogf > 0):
                            other_features = concatenate((other_features, hogf), axis=None)
                        else:
                            ok = False
                    if self.gist:
                        gistf = self.imghp.gist_feature(path)
                        if len(gistf > 0):
                            other_features = concatenate((other_features, gistf), axis=None)
                        else:
                            ok = False
                    if self.deep:
                        deepf = self.imghp.deep_feature(path)
                        if len(deepf > 0):
                            other_features = concatenate((other_features, deepf), axis=None)
                        else:
                            ok = False
                    if ok:
                        if self.sift:
                            sift_imwords.append(sift_features)
                        if self.color or self.hog or self.gist or self.deep:
                            other_imwords.append(other_features)
                        self.testing_paths.append(paths[i])
                if index % 100 == 0 and index > 0:
                    logging.info('Getting image features of %4i out of %i' % (index, len(paths)))
                index += 1
            dt = time() - t0
            all_sift_features = []
            all_other_features = []
            if len(sift_imwords) > 0:
                nparray = array(self.standardize(sift_imwords))
                nbr_occurences = sum((nparray > 0) * 1, axis=0)
                self.idf = log((1.0 * n_images) / (1.0 * nbr_occurences + 1))
                nparray = nparray * self.idf
                all_sift_features = nparray
            if len(other_imwords) > 0:
                all_other_features = array(self.standardize(other_imwords))
            if len(all_sift_features) > 0 and len(all_other_features) > 0:
                self.testing_features = hstack([all_sift_features, all_other_features])
            else:
                if len(all_sift_features) > 0 and len(all_other_features) == 0:
                    self.testing_features = all_sift_features
                if len(all_sift_features) == 0 and len(all_other_features) > 0:
                    self.testing_features = all_other_features
                if len(all_sift_features) == 0 and len(all_other_features) == 0:
                    self.testing_features = []
            logging.info('Done in %.2fs.' % dt)
            logging.info(self.scale.n_samples_seen_)
        return self.testing_features

    def parse_testing_features_in_batch(self, paths):
        logging.info('Extracting features ...')
        self.testing_paths = []
        self.testing_features = []
        if (self.sift) and (self.voc is None):
            logging.info('Done! No vocabulary to build SIFT BOVW features!')
            return [None, None]
        else:
            imwords = []
            t0 = time()
            for i in range(0, len(paths)):
                path = paths[i]
                # logging.info('Loading ' + path)
                # print('Loading ' + path)
                if self.imghp.is_valid_image(path):
                    ok = True
                    features = []
                    if self.sift:
                        kp, desc = self.imghp.sift_feature(path)
                        if len(kp) > 0:
                            data = desc
                            features = concatenate((features, self.project(data)), axis=None)
                        else:
                            ok = False
                    if self.color:
                        colorf = self.imghp.color_feature(path)
                        if len(colorf > 0):
                            features = concatenate((features, colorf), axis=None)
                        else:
                            ok = False
                    if self.hog:
                        hogf = self.imghp.hog_feature(path, 80, 80)
                        if len(hogf > 0):
                            features = concatenate((features, hogf), axis=None)
                        else:
                            ok = False
                    if self.gist:
                        gistf = self.imghp.gist_feature(path)
                        if len(gistf > 0):
                            features = concatenate((features, gistf), axis=None)
                        else:
                            ok = False
                    if self.deep:
                        deepf = self.imghp.deep_feature(path)
                        if len(deepf > 0):
                            features = concatenate((features, deepf), axis=None)
                        else:
                            ok = False
                    if ok:
                        imwords.append(features)
                        self.testing_paths.append(paths[i])
                if i % 100 == 0 and i > 0:
                    logging.info('Getting image features of %4i out of %i' % (i, len(paths)))
            if len(imwords) > 0:
                self.testing_features = array(imwords)
                self.scaler.partial_fit(self.testing_features)
            else:
                self.testing_features = []
                self.testing_paths = []
            dt = time() - t0
            logging.info('Done in %.2fs.' % dt)
        return self.testing_features

    def scale_testing_features_in_batch(self, imwords):
        logging.info('Scaling features ...')
        t0 = time()
        self.testing_features = []
        if len(imwords) > 0:
            self.testing_features = array(self.scaler.transform(imwords))
        else:
            self.testing_features = []
        dt = time() - t0
        logging.info('Done in %.2fs.' % dt)

    def standardize(self, histogram):
        # Standardize features by removing the mean and scaling to unit variance
        self.scale = StandardScaler().fit(histogram)
        return self.scale.transform(histogram)

    def set_scale(self, scale):
        self.scale = scale


class SVMHelper:

    def __init__(self, kernel_type='rbf'):
        self.param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                           'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0], }
        self.clf = GridSearchCV(SVC(kernel=kernel_type, class_weight='balanced', probability=True, random_state=10), self.param_grid,
                                cv=2, verbose=True)

    def train(self, trainning_docs, training_classes):
        t0 = time()
        logging.info("Fitting the SVM classifier to the training set")
        self.clf = self.clf.fit(trainning_docs, training_classes)
        logging.info("Done in %0.3fs" % (time() - t0))
        logging.info("Best estimator found by grid search:")
        logging.info(self.clf.best_estimator_)

    def test(self, testing_docs, testing_classes):
        logging.info("Predicting image class on the test set using SVM")
        t0 = time()
        y_pred = self.clf.best_estimator_.score(testing_docs, testing_classes)
        logging.info("Done in %0.3fs" % (time() - t0))
        return y_pred
