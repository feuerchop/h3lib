#!/usr/bin/env python
# coding: utf-8

import json
import urllib
import sqlite3
import re

# open DB
try:
    dbconn = sqlite3.connect('../h3db/AIAcademicBase.sqlite3')
    dbcur = dbconn.cursor()
except sqlite3.Error as e:
    print "[h3err] An error is occured: ", e.args[0]

# create table icml_authors if not exists
dbcur.execute('''
    CREATE TABLE IF NOT EXISTS nips_authors(
        name TEXT NOT NULL,
        affiliation TEXT DEFAULT 'N/A',
        country TEXT DEFAULT 'N/A',
        email TEXT,
        n_appeared INTEGER DEFAULT 1,
        avc_position REAL) ''')
dbcur.execute(''' CREATE INDEX IF NOT EXISTS idx_author on nips_authors(name) ''')

# source_icml: kimono crawled data hosted on firebase
results_nips11 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/7dcecg3s/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_nips12 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/5qyt8h4c/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_nips13 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/dif28pwm/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_nips14 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/bk7nfs6a/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_nips15 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/24mdcuum/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))

# save new authors found in data source
pattern = re.compile(r"\s[A-Z]{1}[.]\s")
for row in results_nips15[u'results'][u'collection1']:
    author = row[u'nips15_authors'][u'text'].replace('\n', '').strip().encode('utf8').decode('utf8')
    if pattern.search(author):
        print author,'=>',pattern.sub(' ', author)
        author = pattern.sub(' ', author)

    if len(author) > 0:
        dbcur.execute(''' select name, n_appeared from nips_authors where name = ? COLLATE NOCASE''',
                      (author,))
        res = dbcur.fetchone()
        if res is not None:
            cnt, n_appeared = res[0], res[1]
            # author exists, appeared+1, update avc_position
            dbcur.execute('update nips_authors set n_appeared = ? where name = ? COLLATE NOCASE',
                          (n_appeared+1, author,))
        else:
            # new author
            author_capitalized = u''
            for a in author.split(' '):
                author_capitalized = author_capitalized+" "+a[0].upper()+a[1:]
            # print author_capitalized
            dbcur.execute('insert into nips_authors (name) values (?)',
                          (author_capitalized.strip(),))
    else:
        pass

dbconn.commit()
# close DB
dbconn.close()
