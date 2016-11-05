from peewee import Model, MySQLDatabase, IntegerField, TextField


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
