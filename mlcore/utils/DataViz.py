"""
This utility helper file is used for visualizing a dataset from .csv file.
The main usage of this file is for easy check how the training dataset looks like,
and to give developers a quick overview of the data they are dealing with.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
import getopt, sys, os.path, importlib
import plot_utils as pltool
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Range1d


class DataViz(object):
   '''
   A class for dataset viewer used to analyse features in data
   set straightforwardly
   '''

   def __init__(self, config):
      '''
      init a dataviz obj for plotting dataset
      :param config: configuration for plotting
      :return: a figure handler of bokeh
      '''

      if config.has_key('dot_size'):
         self.dot_size = config['dot_size']
      else:
         self.dot_size = 4
      if config.has_key('font_size'):
         self.font_size = config['font_size']
      else:
         self.font_size = 12
      if config.has_key('line_width'):
         self.line_width = config['line_width']
      else:
         self.line_width = 2
      if config.has_key('alpha'):
         self.alpha = config['alpha']
      else:
         self.alpha = 0.3
      if config.has_key('colormap'):
         # default 3 colors
         self.colormap = config['colormap']
      else:
         # binary colors
         self.colormap = 'RdYlBu'
      if config.has_key('color'):
         self.color = config['color']
      else:
         self.color = 'navy'
      if config.has_key('width'):
         self.w = config['width']
      else:
         self.w = 480
      if config.has_key('height'):
         self.h = config['height']
      else:
         self.h = 320

      self.binary_colors = ['#4285F4', '#EA4335']

   def feature_scatter1d(self, X):
      f = figure(width=self.w, height=self.h, webgl=True,
                 toolbar_location='above', title='Feature distribution',
                 active_scroll='wheel_zoom')
      n, d = X.shape
      indices = []
      for i in np.arange(d):
         indices.extend((i + 1) * np.ones(n))
      f.circle(np.asarray(indices), X.T.flatten(), color='gray', size=self.dot_size, alpha=self.alpha)
      return f

   def project2d(self, X, y=None, method='pca', g=0.5):
      '''
      Project high-dimensiona data to 2D for visulaization
      using methods e.g., pca, kernel-pca, mds
      :param X: N*d dataset
      :param y: labels/None if not given
      :param method: string in ['pca','kpca','mds']
      :return: projected dataset X_project
      '''
      if y is not None and y.size != X.shape[0]:
         exit("Data dims are not matched!")
      else:
         n_comp = 2
         if method == 'pca':
            projector = PCA(n_components=n_comp)
            X_proj = projector.fit_transform(X)
         elif method == 'kpca':
            projector = KernelPCA(n_components=n_comp, kernel='rbf', gamma=g)
            X_proj = projector.fit_transform(X)
         elif method == 'mds':
            projector = MDS(n_components=n_comp)
            X_proj = projector.fit_transform(X)
         else:
            print 'No projector found!'
            X_proj = X

      f = figure(width=self.w, height=self.h, webgl=True, toolbar_location='above',
                 title='Sample distribution after ' + method.upper(),
                 active_scroll='wheel_zoom')
      if y is None:
         f.circle(X_proj[:, 0], X_proj[:, 1], color='navy', size=self.dot_size)
      else:
         colors = getattr(importlib.import_module('bokeh.palettes'), self.colormap + str(np.unique(y).size))
         f.circle(X_proj[:, 0], X_proj[:, 1], line_color = None, size=self.dot_size,
                  fill_color=[colors[np.where(np.unique(y) == label)[0]] for label in y])
      return f

   def plot_corr(self, X, names=None):
      n, d = X.shape
      xcorr = np.corrcoef(X.T)
      XX, YY = np.meshgrid(np.arange(d), np.arange(d))
      f = figure(width=self.w, height=self.h, webgl=True, toolbar_location='above',
                 title='Feature Correlations',
                 active_scroll='wheel_zoom')
      # a1 = f.circle(XX.ravel(), YY.ravel(),
      #                  s=15,
      #                  c=xcorr.ravel(), cmap='jet', alpha=0.7)
      # plt.gca().set(title='Correlations of features',
      #               xlim=[0, d - 1],
      #               ylim=[0, d - 1])
      # plt.colorbar(ax=plt.gca())
      # if names is not None and len(names) == d:
      #    a1.set_xticklabels(names, rotation=90)
      #    a1.set_yticklabels(names)
      #
      # pltool.setAxSquare(plt.gca())
      # TODO
      pass

   def fill_between(self, xticks, mean, std, title='Error bar plot',
                    legend=None, xlim=None, ylim=None):
      '''
      plot a shaded error bar plot according to mean and std
      :param xticks:
      :param mean:
      :param std:
      :param output:
      :param title:
      :param legend:
      :param xlim:
      :param ylim:
      :return:
      '''
      fig = figure(title=title, webgl=True, toolbar_location='above',
                   width=self.w, height=self.h,
                   active_scroll='wheel_zoom')
      if xlim is not None:
         fig.set(x_range=Range1d(xlim[0], xlim[1]))
      else:
         fig.set(x_range=Range1d(min(xticks), max(xticks)))

      if ylim is not None:
         fig.set(y_range=Range1d(ylim[0], ylim[1]))

      band_x = np.append(xticks, xticks[::-1])
      if type(legend) is list:
         if len(legend) == 2:
            colors = self.binary_colors
         else:
            colors = getattr(importlib.import_module('bokeh.palettes'), self.colormap + str(len(legend)))
         for m, s, c, l in zip(mean, std, colors, legend):
            band_y = np.append(m - s, (m + s)[::-1])
            fig.patch(band_x, band_y, color=c, fill_alpha=self.alpha)
            fig.line(xticks, m, line_width=self.line_width, line_color=c, legend=l)
            fig.circle(xticks, m, size=self.dot_size, color=c)
      else:
         band_y = np.append(mean - std, (mean + std)[::-1])
         fig.patch(band_x, band_y, color=self.color, fill_alpha=self.alpha)
         fig.line(xticks, mean, line_width=self.line_width, line_color=self.color, legend=legend)
         fig.circle(xticks, mean, size=self.dot_size, color=self.color)

      return fig


if __name__ == '__main__':
   pass
