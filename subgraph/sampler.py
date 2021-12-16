import numpy as np
import networkx as nx
import scipy
import random

from subgraph.graphs import TUDatasetGraph

class OnTheFlySubgraphSampler(object):
  """
    Randomly sample subgraph pairs 
  """
  def __init__(self,av,graphs):
    """
    """
    self.graphs = graphs
    self.graphs_dist = self.generate_graph_dist()
    self.min_subgraph_size = av.MIN_SUBGRAPH_SIZE
    self.max_subgraph_size = av.MAX_SUBGRAPH_SIZE
    #self.pos_to_neg_ratio = 1

  def generate_graph_dist(self):
    """
      Generates a distribution according to which graphs are selected from list of graphs
      Here, graphs are selected proportional to the no. of nodes
    """
    ps = np.array([len(g) for g in self.graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = scipy.stats.rv_discrete(values=(np.arange(len(self.graphs)), ps))
    return dist

  def sample_subgraph(self):
    """
      Returns a randomly selected (connected) list of nodes constituting a subgraph
      and the anchor node and graph_id (to id source graph from a list of graphs)
    """
    while True:
      #Select random graph from list of graphs
      graph_id = self.graphs_dist.rvs()
      graph = self.graphs[graph_id]
      #pick random anchor node
      anchor = random.randint(0,graph.number_of_nodes()-1)
      #pick size of subgraph to be generated
      size = random.randint(self.min_subgraph_size+1,self.max_subgraph_size)
      #init subgraph with anchor node
      subgraph = {anchor}
      #bfs_neigh is set of nodes we consider for adding to subgraph in future
      bfs_neigh = set(graph.neighbors(anchor)) - subgraph
      while len(subgraph)<size and bfs_neigh:
        curr_node = random.choice(list(bfs_neigh))
        subgraph.add(curr_node)
        bfs_neigh.remove(curr_node)
        bfs_neigh.update(set(graph.neighbors(curr_node))-subgraph)
      #if condition is not satisfied go through entire sungraph generation procedure again
      if len(subgraph)>self.min_subgraph_size:
        return graph.subgraph(list(subgraph)),anchor,graph_id

