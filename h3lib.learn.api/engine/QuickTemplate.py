"""
QuickTemplate is the classifier class for tempalte suggestion in MyriadHub. It subclasses MyClassifier and is
instantiated with a proper preprocessor. The classifier uses SGD logistic regression for multi-class classification.

For more details on how to use SGD logistic regression and its corresponding parameters, please checkout:
http://scikit-learn.org/stable/modules/sgd.html

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

from engine.Preprocessor import *
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from engine.MyClassifier import MyClassifier


class QuickTemplate(MyClassifier):
   """
   QuickTempalte main class: a smart template recommendation system used to facilitate easy email reply.
   We are using stochastic gradient descent logistic regression for prediction.
   """

   def __init__(self,
                preprocessor=None,
                loss='log',
                penalty='l2',
                alpha=0.0001,
                shuffle=True,
                class_weight=None,
                max_epoch=10,
                verbose=True):
      """
      Constructor
      :return:
      """

      super(QuickTemplate, self).__init__('Template Suggesting Model', max_epoch=max_epoch, verbose=verbose)
      # training params / optimal training params after model selection
      self.loss = loss
      self.penalty = penalty
      self.alpha = alpha
      self.shuffle = shuffle
      if preprocessor is None:
         # the default preprocessor is a Tfidf vectorizer
         self.preprocessor = Preprocessor(pipeline=[('TfidfVectorizer', {'encoding': 'utf-8'})])
      else:
         self.preprocessor = preprocessor
      # default classifier is SGD logistic regressor
      self.clf = SGDClassifier(loss=self.loss,
                               penalty=self.penalty,
                               alpha=self.alpha,
                               shuffle=self.shuffle,
                               class_weight=class_weight)
      # if verbose:
      #     print '[INFO][{:s}] {:s} is initialized.'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      #                                                  self.name)

   def train(self, trainset, labels=None):
      """
      Training method for build the recommendation system model
      :param trainset: Nxd training data vectors
      """

      self.clf.fit(trainset, labels)
      self.status = 'trained'
      super(QuickTemplate, self).train(trainset, labels)

   def update(self, partial_trainset, labels=None, classes=None):
      """
      Partial training method to dynamically factor in new data into the current model
      :param partial_trainset: Data vectors for subset of data to be trained
      :param labels: Output vector for training data
      :param classes: Array of unique outputs
      :return:
      """
      self.clf.partial_fit(partial_trainset, labels, classes)
      self.status = 'trained'
      super(QuickTemplate, self).update(partial_trainset, labels)

   def predict(self, testset):
      """
      Classification of which template to use for replying an email
      :param testset: Mxd test data vectors
      :return: Template labels, e.g., 1,2,..., n (integer)
      """

      if self.status is 'untrained':
         print 'classifier is not trained yet! exit..'
         return False
      return self.clf.predict(testset)

   def model_selection(self, trainset, labels):
      """
      Model selection to optimize over traning set to compute which optimal parameters for training should be used.
      :param data_raw: raw data
      """

      # generates 10 values in range 10^-1 to 10^-7
      alpha_s = 10.0 ** -np.arange(1, 7, 10)
      clf = GridSearchCV(estimator=self.clf, param_grid=dict(alpha=alpha_s), n_jobs=-1, verbose=1)
      clf.fit(trainset, labels)
      self.clf = clf.best_estimator_
      self.status = 'trained'
      return clf.best_estimator_.alpha, clf.best_score_

   def plot_classifier(self, savepath=None, show=True):
      '''
      plotting ultilities for QuickTemplate
      :param: savepath: the path to save the figure HTML files
      :return: NoneType
      '''

      from bokeh.plotting import output_file, figure, show, ColumnDataSource, save
      from bokeh.layouts import gridplot
      from bokeh.models import FixedTicker, HoverTool, BoxZoomTool, ResetTool, WheelZoomTool
      from sklearn.preprocessing import minmax_scale

      output_file(savepath, title='QuickTempalte Classifier Summary')

      figs_row = []
      ticks = []
      tick_labels = []
      feature_idx = 0
      for p in self.preprocessor:
         if p._FEATURE_SIZE:
            if p._FEATURE_SIZE != len(p._FEATURE_NAMES):
               ticks.extend([p._FEATURE_SIZE / 2])
            else:
               ticks.extend(range(feature_idx, feature_idx + p._FEATURE_SIZE))
            feature_idx += p._FEATURE_SIZE
            tick_labels.extend(p._FEATURE_NAMES)
      # TOOLS = [HoverTool()]
      cls_id = 0
      for cls in self.clf.classes_:
         cls_name = str(cls)
         if ticks[0] > 0:
            y_ticks = np.r_[np.array([self.clf.coef_[cls_id][ticks[0]]]),
                            self.clf.coef_[cls_id][ticks[1]:]].ravel()
         else:
            y_ticks = self.clf.coef_[cls_id].ravel()
         y_ticks = minmax_scale(y_ticks, feature_range=(-1, 1))
         source = ColumnDataSource(data=dict(
            x=ticks,
            y=y_ticks / 2.0,
            v=y_ticks,
            names=tick_labels
         ))
         hover = HoverTool(tooltips=[("feature name", "@names"),
                                     ("value", "@v")])
         fig = figure(width=960, height=180,
                      title='Feature Relevance for Tempalte-ID: ' + cls_name,
                      tools=[hover, BoxZoomTool(), ResetTool(), WheelZoomTool()])
         fig.rect(x='x',
                  y='y',
                  width=.5,
                  height=y_ticks,
                  color='#CAB2D6',
                  source=source)

         fig.xaxis[0].ticker = FixedTicker(ticks=ticks)
         fig.xaxis.major_label_orientation = np.pi / 4
         figs_row.append([fig])
         cls_id += 1
      # show(gridplot(figs_row))
      save(gridplot(figs_row), savepath)


if __name__ == '__main__':
   """
   User Guide
   """

   import matplotlib.pyplot as plt
   from sklearn.datasets import load_digits
   from sklearn.cross_validation import train_test_split
   from sklearn.metrics import accuracy_score
   from sklearn.metrics import f1_score
   from sklearn.metrics import precision_score
   from sklearn.metrics import recall_score

   # Load data
   # Normally we use Xtr denoting training feature set, and ytr denoting training labels
   mydata = load_digits()
   Xtr = mydata.data
   ytr = mydata.target
   myTemplate = QuickTemplate()
   # split the dataset as train/test
   X_train, X_test, y_train, y_test = train_test_split(Xtr, ytr, test_size=0.33, random_state=30)

   fig = plt.figure()
   ax = fig.add_subplot(111)

   epoch = 1
   batch_size = 20
   steps = 100
   tr_size = X_train.shape[0]
   for n in range(epoch):
      # divisions = 1     #number of partitions of data (how many batches of data will be trained)
      accuracies = []

      for i in xrange(steps):
         batch_start = i * batch_size % tr_size
         batch_end = min(batch_start + batch_size, tr_size)
         X_train_part = X_train[batch_start:batch_end]
         y_train_part = y_train[batch_start:batch_end]

         # Partial train the segment of data, classify test data and compare to actual values using various data analysis methods
         myTemplate.update(X_train_part, labels=y_train_part, classes=np.unique(ytr))
         y_pred = myTemplate.predict(X_test)
         accuracy = 100 * accuracy_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred, average=None)
         precision = precision_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)

         accuracies.append(accuracy)

   ax.plot(xrange(steps), accuracies, 'k-', lw=2)
   ax.set(title='SGD accuracy', xlabel='steps', ylabel='accuracy')
   plt.ylim(0, 100)
   plt.show()
