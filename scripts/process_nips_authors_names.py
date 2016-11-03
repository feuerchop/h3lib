#!/usr/bin/env python
# coding: utf-8

import sqlite3

dbconn = sqlite3.connect('../h3db/AIAcademicBase.sqlite3')
dbcur = dbconn.cursor()

dbcur.execute(''' select name from nips_authors ''')
rows = dbcur.fetchall()
for row in rows:
    print row

dbconn.close()