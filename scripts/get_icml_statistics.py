#!/usr/bin/env python
# coding: utf-8

import json
import urllib
import sqlite3

# open DB
try:
    dbconn = sqlite3.connect('../h3db/AIAcademicBase.sqlite3')
    dbcur = dbconn.cursor()
except sqlite3.Error as e:
    print "[h3err] An error is occured: ", e.args[0]

# create table icml_authors if not exists
dbcur.execute('''
    CREATE TABLE IF NOT EXISTS icml_authors(
        name TEXT NOT NULL,
        affiliation TEXT DEFAULT 'N/A',
        country TEXT DEFAULT 'N/A',
        email TEXT,
        n_appeared INTEGER DEFAULT 1) ''')
dbcur.execute(''' CREATE INDEX IF NOT EXISTS idx_author on icml_authors(name) ''')

# source_icml: kimono crawled data hosted on firebase
results_icml11 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/9dddmfj6/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_icml12 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/4zpjozlw/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_icml13 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/bpcqdplm/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_icml14 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/3bvqawso/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))
results_icml15 = json.load(urllib.urlopen(
    "https://crawlydb.firebaseio.com//kimono/api/3al38z4q/latest.json?auth=GcbFpZq0UqoadmWMQiKGY0nydNTCjlL25c5dSpL7"))

# icml_exist_authors = dbcur.execute('select name from icml_authors').fetchall()

# save new authors found in data source
for row in results_icml11[u'results'][u'collection1']:
    for idx, author in enumerate(row[u'icml11_authors'].replace('\n', '').split(',')):
        author = author.strip().encode('utf8').decode('utf8')
        if len(author) > 0:
            dbcur.execute(''' select name, n_appeared, avc_position from icml_authors where name = ? COLLATE NOCASE''',
                          (author,))
            res = dbcur.fetchone()
            if res is not None:
                cnt, n_appeared, avc_pos = res[0], res[1], res[2]
                # author exists, appeared+1, update avc_position
                dbcur.execute('update icml_authors set n_appeared = ?, avc_position=? where name = ? COLLATE NOCASE',
                              (n_appeared+1, n_appeared*avc_pos/(n_appeared+1)+1.0*(idx+1)/(n_appeared+1), author,))
            else:
                # new author
                author_capitalized = u''
                for a in author.split(' '):
                    author_capitalized = author_capitalized+" "+a[0].upper()+a[1:]
                print author_capitalized
                dbcur.execute('insert into icml_authors (name, avc_position) values (?, ?)',
                              (author_capitalized.strip(), idx+1.0))
        else:
            pass

dbconn.commit()
# close DB
dbconn.close()
