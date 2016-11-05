"""
Base class for machine learning classifier. It is an abstract class defining
methods need to be implemented in subclasses.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

from abc import ABCMeta, abstractmethod
from datetime import datetime
import dill
import numpy as np

class MyClassifier(object):

    '''
    Abstract class for classifiers of SmartEngine.
    All smart components should inherit from this super class.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, name='MyClassifier', preprocessor=None, max_epoch=10, verbose=False):
        '''
        Initialization
        :param name: the name of the classifier
        :param preprocessor: a list of preprocessors
        :param max_epoch: maximal epochs to train
        :param verbose: verbose print out information
        '''
        self.name = name
        self.max_epoch = max_epoch
        self.status = 'untrained'
        self.preprocessor = preprocessor
        self.verbose = verbose

    @abstractmethod
    def train(self, featureset, labelset):
        # # train on a training dataset
        # if self.verbose:
        #     print '\t\t[INFO][{:s}] classifier is trained on {:d} samples ...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #                                                                       featureset.shape[0])
        pass

    @abstractmethod
    def update(self, minibatch_featureset, minibatch_labelset):
        # update model on a minibatch
        # if self.verbose:
        #     print '\t\t[INFO][{:s}] classifier is updated on {:d} samples ...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #                                                                       minibatch_featureset.shape[0])
        pass

    @abstractmethod
    def predict(self, test_featureset):
        # predict outputs for test dataset
        pass

    def save(self, path):
        # save checkpoint for the predictive model
        dill.dump(self, open(path, 'w'))
        if self.verbose:
            print '[INFO][{:s}] classifier checkpoint is saved at {:s} ...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                    path)

    def add_preprocessor(self, pc):
        '''
        Append additional preprocessor to the list of preprocessor in this classifier.
        :param pc: an instance of preprocessor
        :return:
        '''
        # append a new preprocessor
        self.preprocessor.append(pc)

    def prep_data(self, data_blocks, restart=False):
        '''
        prepare a trainable dataset from a list data blocks each of which is processable
        by its preprocessor accordingly. Processed data blocks are concatenated as a bigger trainable dataset.
        :param data_blocks: a list of data blocks
        :return: A nxd trainable ndarray, d = sum(feature sizes of data blocks)
        '''

        begin = True
        if self.preprocessor is not None:
            nrows = 0
            if type(self.preprocessor) is not list:
                self.preprocessor = [self.preprocessor]
            if len(self.preprocessor) != len(data_blocks):
                print 'num. of data blocks do not align with num. of preprocessors in classifer.'
            for pc, block in zip(self.preprocessor, data_blocks):
                if len(block) == 0:
                    # empty data block
                    pc._FEATURE_NAMES = []
                    pc._FEATURE_SIZE = 0
                    continue
                if begin:
                    output = pc.run(block, restart=restart)
                    nrows = output.shape[0]
                    begin = False
                else:
                    cur_output = pc.run(block, restart=restart)
                    if cur_output.shape[0] != nrows:
                        print 'Output of preprocessor does not align with previous data block.'
                        return False
                    else:
                        output = np.c_[output, cur_output]
            return output
        else:
            print 'No preprocess is found in this classifier... return False'
            return False

    @abstractmethod
    def plot_classifier(self, **kwargs):
        '''
        Implement the plotting function in corresponding classifier class.
        '''
        pass