'''
A Java snippet security detector using TfIdf
'''
import csv, sys, datetime as dt, argparse
import numpy as np
from peewee import *

sys.path.append('../')
from engine.QuickTemplate import QuickTemplate
from engine.Preprocessor import Preprocessor
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


class JavaSnippets(Model):
   '''
    Data Model for JavaSnippet Instance
    '''

   class Meta:
      database = MySQLDatabase('java_snippets_sec_db', host="localhost", user="root", passwd="damnshit")

   snippet_id = IntegerField()
   snippet = TextField(null=True)
   true_sec_level = IntegerField(null=True, default=0)
   predict_sec_level = IntegerField(null=True, default=0)

db = JavaSnippets._meta.database
db.connect()
db.create_tables([JavaSnippets], safe=True)

argparser = argparse.ArgumentParser()
argparser.add_argument("--filename", help="input snippet csv file path...")
args = argparser.parse_args()
raw_data_path = args.filename

with open(raw_data_path, 'rb') as csv_file:
   print 'Cross validation on file: {:s}'.format(raw_data_path)
   samples = csv.reader(csv_file, delimiter=',')
   X, y = list(), list()
   for id, sample in enumerate(samples):
      X.append(sample[0])
      y.append(int(sample[1]))
      if id > 0 and id % 100 == 0:
         print '{:d} samples are processed ... '.format(id)
      if JavaSnippets.select().where(JavaSnippets.snippet_id == id).count() == 0:
         # insert new row
         snippet = JavaSnippets()
         snippet.snippet_id = id
         snippet.snippet = sample[0]
         snippet.true_sec_level = sample[1]
         snippet.save()

# prepare classifier
work_flow = [{'worker': 'TfidfVectorizer', 'params': {'encoding': 'utf-8'}},
             {'worker': 'FeatureScaler', 'params': {'type': 'minmax'}}]
pp = Preprocessor(work_flow)
SnippetClf = QuickTemplate(preprocessor=[pp])
Xtr = SnippetClf.prep_data(data_blocks=[X])
ytr = np.array(y)
cvfolds = StratifiedKFold(ytr, n_folds=5)
fold_id = 0
for tr_idx, tt_idx in cvfolds:
   xtr_fold, xtt_fold = Xtr[tr_idx], Xtr[tt_idx]
   ytr_fold, ytt_fold = ytr[tr_idx], ytr[tt_idx]
   print '[{:d}]-fold: {:d} training samples / {:d} testing samples'.format(fold_id, len(tr_idx), len(tt_idx))
   run_start = dt.datetime.now()
   SnippetClf.train(xtr_fold, ytr_fold)
   y_predict = SnippetClf.predict(xtt_fold)
   acc = accuracy_score(ytt_fold, y_predict)
   run_end = dt.datetime.now()
   print '\t accuracy: {:f}, time spent {:f} msec.'.format(acc, (run_end - run_start).total_seconds())
   fold_id += 1
   for elem in zip(tt_idx, y_predict):
      tid, yt = elem[0], elem[1]
      query = JavaSnippets.update(predict_sec_level=yt).where(JavaSnippets.snippet_id==tid)
      query.execute()

