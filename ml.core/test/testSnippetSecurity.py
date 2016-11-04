'''
A Java snippet security detector using TfIdf
'''
import csv, datetime
from ..engine import PipelineWorkers, Preprocessor, QuickTemplate

# process input data
raw_data_path = '~/repos/h3lib/h3db/snippet_data.csv'
with open(raw_data_path, 'rb') as csv_file:
   samples = csv.reader(csv_file, delimiter=',')
   X, y = list(), list()
   for id, sample in enumerate(samples):
      X.append(sample[0])
      y.append(int(sample[1]))
      if id % 100 == 0:
         print '{:d} samples are processed ... '.format(id)

# prepare classifier
work_flow = [{'worker': 'TfIdfVectorizer', 'params': {'encoding': 'utf-8'}},
             {'worker': 'FeatureScaler', 'params': {'type': 'minmax'}}]
pp = Preprocessor.Preprocessor(work_flow)
SnippetClf = QuickTemplate.QuickTemplate(preprocessor=[pp])


