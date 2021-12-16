import numpy as np
import networkx as nx
from collections import defaultdict
from common import logger
import pickle
import os

from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils

class TUDatasetGraph(object):
  """
    Source link - https://chrsmrrs.github.io/datasets/docs/datasets/
    Each dataset is a set of graphs.
  """
  def __init__(self,av):
    """
    """
    self.av = av
    self.dataset_name = self.av.DATASET_NAME
    self.dataset = self.load_dataset()

  def load_dataset(self):
    """
      Downloads dataset the first time and stores as pkl. 
      Subsequently, dataset is loaded directly from pkl
    """
    fname = self.av.DIR_PATH + "/Datasets/"+self.dataset_name+".pkl"
    if os.path.isfile(fname):
      d = pickle.load(open(fname,"rb"))
    else:
      if self.dataset_name == "enzymes":
        d = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
      elif self.dataset_name == "proteins":
        d = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
      elif self.dataset_name == "cox2":
        d = TUDataset(root="/tmp/cox2", name="COX2")
      elif self.dataset_name == "aids":
        d = TUDataset(root="/tmp/AIDS", name="AIDS")
      elif self.dataset_name == "mutag":
        d = TUDataset(root="/tmp/MUTAG", name="MUTAG")
      elif self.dataset_name == "ptc_fm":
        d = TUDataset(root="/tmp/PTC_FM", name="PTC_FM")        
      elif self.dataset_name == "ptc_fr":
        d = TUDataset(root="/tmp/PTC_FR", name="PTC_FR")   
      elif self.dataset_name == "ptc_mm":
        d = TUDataset(root="/tmp/PTC_MM", name="PTC_MM")   
      elif self.dataset_name == "ptc_mr":
        d = TUDataset(root="/tmp/PTC_MR", name="PTC_MR")  
      else:
        raise NotImplementedError()
      with open(fname,"wb") as f:
        pickle.dump(d,f)
    return d

  def get_nx_graph(self):
    """
      Convert the list of pyG "Data" graphs to list of nx graphs and return
    """
    d_nx = []
    for g in list(self.dataset):
      g_nx = pyg_utils.to_networkx(g).to_undirected()
      d_nx.append(g_nx)
    return d_nx

  def get_node_labels(self):
    """
      populate list of np.array for list of graphs in dataset
    """
    #TODO: Not all TU datasets have node labels. Add code to check by name
    return [np.array(d.x) for d in self.dataset]

  def get_node_label_dim(self):
    return list(self.dataset)[0].x.shape[1]

  def get_node_features(self):
    """
      populate list of np.array for list of graphs in dataset
    """
    #TODO: DUMMY
    return []

  def get_edge_labels(self):
    """
      populate list of np.array for list of graphs in dataset
    """
    #TODO: DUMMY
    return []

  def get_edge_features(self):
    """
      populate list of np.array for list of graphs in dataset
    """
    #TODO: DUMMY
    return []

