import os, cherrypy
from apps.DialogType import Mails


class BayesNet(object):
   def __init__(self):
      pass

   @cherrypy.expose
   def index(self):
      return 'index... '

   @cherrypy.expose
   def info(self):
      return 'this is info... '


if __name__ == '__main__':
   cherrypy.config.update(config='confs/global.cfg')
   # cherrypy.tree.mount(BayesNet(), '/bayesnet', config='confs/global.cfg')
   cherrypy.tree.mount(Mails(), '/mails', config='confs/DialogType.cfg')
   cherrypy.engine.start()
   cherrypy.engine.block()
