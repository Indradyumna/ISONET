import logging, sys, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def set_log(av):
  if av:
    if not os.path.isdir(os.path.dirname(av.logpath)): 
      os.makedirs(os.path.dirname(av.logpath))    
    handler = logging.FileHandler(av.logpath)
    logger.addHandler(handler)
