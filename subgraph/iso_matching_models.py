import argparse
import sys
import scipy
import torch
import torch.nn as nn
import pickle
from common import logger, set_log
import networkx as nx
import random 
import math
import numpy as np
from subgraph.utils import cudavar,save_initial_model, save_model_at_epoch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import os
from sklearn.metrics import average_precision_score
import time
from subgraph.earlystopping import EarlyStoppingModule
#import copy
from torch.nn.utils.rnn import  pad_sequence 
from subgraph.graphs import TUDatasetGraph

import torch.nn.functional as F
import collections
import GMN.utils as gmnutils
import GMN.graphembeddingnetwork as gmngen
import GMN.graphmatchingnetwork as gmngmn
from GMN.configure import *
from GMN.loss import euclidean_distance
import GraphOTSim.python.layers as gotsim_layers
from lap import lapjv
from lap import lapmod
import subgraph.neuromatch as nm


class OurMatchingModelSubgraphIsoData(object):
  """
  """
  def __init__(self,av,mode="train"):
    self.av = av
    self.mode = mode
    self.load_graphs()
    self.preprocess_subgraphs_to_pyG_data()
    self.fetch_subgraph_adjacency_info()
    self.data_type = "pyg" #"pyg"/"gmn"

  def load_graphs(self):
    """
      self.query_graphs : list of nx graphs 
      self.corpus_graphs : list of nx graphs
      self.rels : dict denoting 'pos'/'neg' 
      self.list_pos
      self.list_neg
    """

    #LOAD query graphs in data split
    if self.mode == "Extra_test_300":
        fp = self.av.DIR_PATH + "/Datasets/splits/Extra_test_300/test_" +self.av.DATASET_NAME +"_80k_query_subgraphs.pkl"
    else:
        fp = self.av.DIR_PATH + "/Datasets/splits/" + self.mode + "/" + self.mode + "_" +\
                self.av.DATASET_NAME +"80k_query_subgraphs.pkl"

    self.query_graphs = pickle.load(open(fp,"rb"))
    logger.info("loading %s query graphs from %s", self.mode, fp)

    #LOAD iso relaitonships of query graphs wrt corpus graphs
    if self.mode == "Extra_test_300":
        fp = self.av.DIR_PATH + "/Datasets/splits/Extra_test_300/test_" +self.av.DATASET_NAME + "_80k_rel_nx_is_subgraph_iso.pkl"
    else:
        fp = self.av.DIR_PATH + "/Datasets/splits/" + self.mode + "/" + self.mode + "_" +\
                self.av.DATASET_NAME + "80k_rel_nx_is_subgraph_iso.pkl"
    self.rels = pickle.load(open(fp,"rb"))
    logger.info("loading %s relationships from %s", self.mode, fp)

    #LOAD all corpus graphs
    fp = self.av.DIR_PATH + "/Datasets/splits/" + self.av.DATASET_NAME +"80k_corpus_subgraphs.pkl"
    self.corpus_graphs = pickle.load(open(fp,"rb"))
    logger.info("loading corpus graphs from %s", fp)
    assert(list(range(len(self.query_graphs))) == list(self.rels.keys()))

    
    self.list_pos = []
    self.list_neg = []   

    for q in range(len(self.query_graphs)) :
      for c in self.rels[q]['pos']:
        self.list_pos.append(((q,c),1.0))
      for c in self.rels[q]['neg']:
        self.list_neg.append(((q,c),0.0))  
        
  def create_pyG_data_object(self,g):
    if self.av.FEAT_TYPE == "One":
      #This sets node features to one aka [1]
      x1 = cudavar(self.av,torch.FloatTensor(torch.ones(g.number_of_nodes(),1)))
    else:
      raise NotImplementedError()  
      
    l = list(g.edges)
    edges_1 = [[x,y] for (x,y) in l ]+ [[y,x] for (x,y) in l]
    edge_index = cudavar(self.av,torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long))
    return Data(x=x1,edge_index=edge_index),g.number_of_nodes()
  
  def preprocess_subgraphs_to_pyG_data(self):
    """
      self.query_graph_data_list
      self.query_graph_size_list
      self.corpus_graph_data_list
      self.corpus_graph_size_list
    """
    self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
    if self.av.FEAT_TYPE == "One":
        self.num_features = 1
    else:
      self.num_features = self.max_set_size
    self.query_graph_data_list = []
    self.query_graph_size_list = []
    n_graphs = len(self.query_graphs)
    for i in range(n_graphs): 
      data,size = self.create_pyG_data_object(self.query_graphs[i])
      self.query_graph_data_list.append(data)
      self.query_graph_size_list.append(size)

    self.corpus_graph_data_list = []
    self.corpus_graph_size_list = []
    n_graphs = len(self.corpus_graphs)
    for i in range(n_graphs): 
      data,size = self.create_pyG_data_object(self.corpus_graphs[i])
      self.corpus_graph_data_list.append(data)
      self.corpus_graph_size_list.append(size)     

  def fetch_subgraph_adjacency_info(self):
    """
      used for input to hinge scoring
      self.query_graph_adj_list
      self.corpus_graph_adj_list
    """
    self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
    self.query_graph_adj_list = []
    n_graphs = len(self.query_graphs)
    for i in range(n_graphs):
      g = self.query_graphs[i]
      x1 = cudavar(self.av,torch.FloatTensor(nx.adjacency_matrix(g).todense()))
      x2 = F.pad(x1,pad=(0,self.max_set_size-x1.shape[1],0,self.max_set_size-x1.shape[0]))
      self.query_graph_adj_list.append(x2)

    self.corpus_graph_adj_list = []
    n_graphs = len(self.corpus_graphs)
    for i in range(n_graphs): 
      g = self.corpus_graphs[i]
      x1 = cudavar(self.av,torch.FloatTensor(nx.adjacency_matrix(g).todense()))
      x2 = F.pad(x1,pad=(0,self.max_set_size-x1.shape[1],0,self.max_set_size-x1.shape[0]))
      self.corpus_graph_adj_list.append(x2)      

  def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
    Args:
      graphs: a list of generated networkx graphs.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features
            node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )    
    
  def create_stratified_batches(self):
    """
      Creates shuffled batches while maintaining class ratio
    """
    lpos = self.list_pos
    lneg = self.list_neg
    random.shuffle(lpos)
    random.shuffle(lneg)
    p2n_ratio = len(lpos)/len(lneg)
    npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.av.BATCH_SIZE)
    nneg = self.av.BATCH_SIZE-npos
    batches_pos, batches_neg = [],[]
    for i in range(0, len(lpos), npos):
      batches_pos.append(lpos[i:i+npos])
    for i in range(0, len(lneg), nneg):
      batches_neg.append(lneg[i:i+nneg])
     
    self.num_batches = min(len(batches_pos),len(batches_neg))  
    self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
    return self.num_batches

  def create_batches(self,list_all):
    """
      create batches as is and return number of batches created
    """

    self.batches = []
    for i in range(0, len(list_all), self.av.BATCH_SIZE):
      self.batches.append(list_all[i:i+self.av.BATCH_SIZE])
   
    self.num_batches = len(self.batches)  

    return self.num_batches
        
  def fetch_batched_data_by_id(self,i):
    """
      all_data  : graph node, edge info
      all_sizes : this is required to create padding tensors for
                  batching variable size graphs
      target    : labels/scores             
    """
    assert(i < self.num_batches)  
    batch = self.batches[i]
    
    a,b = zip(*batch)
    g_pair = list(a)
    score = list(b)
    
    a,b = zip(*g_pair)
    if self.data_type =="gmn":
      g1 = [self.query_graphs[i] for i in a]  
    else:
      g1 = [self.query_graph_data_list[i] for i in a]
    g1_size = [self.query_graph_size_list[i] for i in a]
    g1_adj  = [self.query_graph_adj_list[i]  for i in a]
    if self.data_type =="gmn":
      g2 = [self.corpus_graphs[i] for i in b]      
    else:    
      g2 = [self.corpus_graph_data_list[i] for i in b]
    g2_size = [self.corpus_graph_size_list[i] for i in b]
    g2_adj  = [self.corpus_graph_adj_list[i]  for i in b]
    
    if self.data_type =="gmn":
      all_data = self._pack_batch(zip(g1,g2))
    else:
      all_data = list(zip(g1,g2))
    all_sizes = list(zip(g1_size,g2_size))
    all_adj = list(zip(g1_adj,g2_adj))
    target = cudavar(self.av,torch.tensor(score))
    return all_data, all_sizes, target, all_adj



def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = cudavar(av,torch.rand(shape).float())
  return -torch.log(eps - torch.log(U + eps))

def pytorch_sinkhorn_iters(av, log_alpha,temp=0.1,noise_factor=1.0, n_iters = 20):
    noise_factor = av.NOISE_FACTOR
    batch_size = log_alpha.size()[0]
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    noise = pytorch_sample_gumbel(av,[batch_size, n, n])*noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha,temp)

    for i in range(n_iters):
      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)

      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
    return torch.exp(log_alpha)

class GMN_match(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_match, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -euclidean_distance(x,y)
        
        return scores

class SimGNN(torch.nn.Module):
    def __init__(self, av,input_dim):
        """
        """
        super(SimGNN, self).__init__()
        self.av = av
        self.input_dim = input_dim

        #Conv layers
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.av.filters_1)
        self.conv2 = pyg_nn.GCNConv(self.av.filters_1, self.av.filters_2)
        self.conv3 = pyg_nn.GCNConv(self.av.filters_2, self.av.filters_3)
        
        #Attention
        self.attention_layer = torch.nn.Linear(self.av.filters_3,self.av.filters_3, bias=False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        #NTN
        self.ntn_a = torch.nn.Bilinear(self.av.filters_3,self.av.filters_3,self.av.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_a.weight)
        self.ntn_b = torch.nn.Linear(2*self.av.filters_3,self.av.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_b.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(self.av.tensor_neurons,1))
        torch.nn.init.xavier_uniform_(self.ntn_bias)
        #Final FC
        feature_count = (self.av.tensor_neurons+self.av.bins) if self.av.histogram else self.av.tensor_neurons
        self.fc1 = torch.nn.Linear(feature_count, self.av.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.av.bottle_neck_neurons, 1)

    def GNN (self, data):
        """
        """
        features = self.conv1(data.x,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        return features

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        query_batch.x = self.GNN(query_batch)
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]
        
        corpus_batch = Batch.from_data_list(c_graphs)
        corpus_batch.x = self.GNN(corpus_batch)
        corpus_gnode_embeds = [g.x for g in corpus_batch.to_data_list()]

        preds = []
        q = pad_sequence(query_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()

        c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(c),dim=1).T,cgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(c@context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()
        
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())
        

        #TODO: Figure out how to tensorize this
        if self.av.histogram == True:
          h = torch.histc(q@c.permute(0,2,1),bins=self.av.bins)
          h = h/torch.sum(h)

          scores = torch.cat((scores, h),dim=1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds.append(score)
        p = torch.stack(preds).squeeze()
        return p

class Node_align_Node_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Node_align_Node_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        
   

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        a, b = zip(*batch_adj)
        q_adj = torch.stack(a)
        c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], \
                        [x.shape[0] for x in node_feature_enc_corpus])) \
               == batch_data_sizes)        
        
        
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])


        transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)
 
        sinkhorn_input = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
        
        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        return scores


class GMN_embed(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        
   

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -euclidean_distance(x,y)
        
        return scores

class ISONET(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(ISONET, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

        
    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
        #and no. of nodes is not equal to no. of edges
        #so a hack to obtain no of edges in each graph from available info
        from GMN.segment import unsorted_segment_sum
        tt = unsorted_segment_sum(cudavar(self.av,torch.ones(len(to_idx))), to_idx, len(graph_idx))
        tt1 = unsorted_segment_sum(cudavar(self.av,torch.ones(len(from_idx))), from_idx, len(graph_idx))
        edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
        edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
        assert(edge_counts == edge_counts1).all()
        assert(sum(edge_counts)== len(to_idx))
        return list(map(int,edge_counts.tolist()))

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        #self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_features_enc = forward_edge_msg + backward_edge_msg
        
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        qgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[0::2]))
        cgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[1::2]))

        edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]  
        
        
        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])


        transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
        transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes]))
        masked_qedge_emb = torch.mul(qgraph_mask,transformed_qedge_emb)
        masked_cedge_emb = torch.mul(cgraph_mask,transformed_cedge_emb)
 
        sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        return scores

class Node_align_Edge_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Node_align_Edge_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

        
   

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    
    def compute_edge_scores_from_node_embeds(self,H, H_sz): 
        """
          H    : (batched) node embedding matrix with each row a node embed
          H_sz : garh sizes of all graphs in the batch
        """
        #we want all pair combination of node embeds
        #repeat and repeat_interleave H and designate either of them as source and the other destination 
        source = torch.repeat_interleave(H,repeats=self.max_set_size,dim =1)
        destination =  H.repeat(1,self.max_set_size,1)
        #each edge feature is [1] 
        edge_emb = cudavar(self.av,torch.ones(source.shape[0],source.shape[1],1))
        #Undirected graphs - hence do both forward and backward concat for each edge 
        forward_batch = torch.cat((source,destination,edge_emb),dim=-1)
        backward_batch = torch.cat((destination,source,edge_emb),dim=-1)
        #use message encoding network from GMN encoding to obtain forward and backward score for each edge
        forward_msg_batch = self.edge_score_fc(self.prop_layer._message_net(forward_batch))
        backward_msg_batch = self.edge_score_fc(self.prop_layer._reverse_message_net(backward_batch))
        #design choice to add forward and backward scores to get total edge score
        bidirectional_msg_batch = torch.cat((forward_msg_batch,backward_msg_batch),dim=-1)
        #note the reshape here to get M matrix
        edge_scores_batch = torch.sum(bidirectional_msg_batch,dim=-1).reshape(-1,self.max_set_size,self.max_set_size)
        #mask the rows and cols denoting edges with dummy node either side
        mask_batch = cudavar(self.av,torch.stack([self.set_size_to_mask_map[i] for i in H_sz]))
        masked_edge_scores_batch = torch.mul(edge_scores_batch,mask_batch)    
        #TODO: NOTE: May need to fill diagonal with 0
        return masked_edge_scores_batch

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        a, b = zip(*batch_adj)
        q_adj = torch.stack(a)
        c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], \
                        [x.shape[0] for x in node_feature_enc_corpus])) \
               == batch_data_sizes)        
        
        
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])


        transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)
 
        sinkhorn_input = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
 
        qgr_edge_scores_batch = self.compute_edge_scores_from_node_embeds(stacked_qnode_emb,qgraph_sizes)
        cgr_edge_scores_batch = self.compute_edge_scores_from_node_embeds(stacked_cnode_emb,cgraph_sizes)
       
        qgr_edge_scores_batch_masked = torch.mul(qgr_edge_scores_batch,q_adj)
        cgr_edge_scores_batch_masked = torch.mul(cgr_edge_scores_batch,c_adj)
        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(qgr_edge_scores_batch_masked - transport_plan@cgr_edge_scores_batch_masked@transport_plan.permute(0,2,1),\
               cudavar(self.av,torch.tensor([0]))),\
            dim=(1,2))
        
        return scores



class GMN_embed_hinge(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed_hinge, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        


    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        #scores = -euclidean_distance(x,y)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

    
class GMN_match_hinge(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_match_hinge, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        #scores = -euclidean_distance(x,y)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

class ISONET_Sym(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(ISONET_Sym, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

        
    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
        #and no. of nodes is not equal to no. of edges
        #so a hack to obtain no of edges in each graph from available info
        from GMN.segment import unsorted_segment_sum
        tt = unsorted_segment_sum(cudavar(self.av,torch.ones(len(to_idx))), to_idx, len(graph_idx))
        tt1 = unsorted_segment_sum(cudavar(self.av,torch.ones(len(from_idx))), from_idx, len(graph_idx))
        edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
        edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
        assert(edge_counts == edge_counts1).all()
        assert(sum(edge_counts)== len(to_idx))
        return list(map(int,edge_counts.tolist()))

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        #self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_features_enc = forward_edge_msg + backward_edge_msg
        
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        qgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[0::2]))
        cgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[1::2]))

        edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]  
        
        
        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])


        transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
        transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes]))
        masked_qedge_emb = torch.mul(qgraph_mask,transformed_qedge_emb)
        masked_cedge_emb = torch.mul(cgraph_mask,transformed_cedge_emb)
 
        sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum((stacked_qedge_emb - transport_plan@stacked_cedge_emb) ** 2, dim=(1,2))
        
        return scores


class GraphSim(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(GraphSim, self).__init__()
        self.av = av
        self.config = config
        if self.av.FEAT_TYPE == "Onehot1":
          self.input_dim = max(self.av.MAX_CORPUS_SUBGRAPH_SIZE, self.av.MAX_QUERY_SUBGRAPH_SIZE)
        else:
          self.input_dim = input_dim
        self.build_layers()

    def build_layers(self):

        self.gcn_layers = torch.nn.ModuleList([])
        self.conv_layers = torch.nn.ModuleList([])
        self.pool_layers = torch.nn.ModuleList([])
        self.linear_layers = torch.nn.ModuleList([])
        self.num_conv_layers = len(self.config['graphsim']['conv_kernel_size'])
        self.num_linear_layers = len(self.config['graphsim']['linear_size'])
        self.num_gcn_layers = len(self.config['graphsim']['gcn_size'])

        num_ftrs = self.input_dim
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                pyg_nn.GCNConv(num_ftrs, self.config['graphsim']['gcn_size'][i]))
            num_ftrs = self.config['graphsim']['gcn_size'][i]

        in_channels = 1
        for i in range(self.num_conv_layers):
            self.conv_layers.append(gotsim_layers.CNNLayerV1(kernel_size=self.config['graphsim']['conv_kernel_size'][i],
                stride=1, in_channels=in_channels, out_channels=self.config['graphsim']['conv_out_channels'][i],
                num_similarity_matrices=self.num_gcn_layers))
            self.pool_layers.append(gotsim_layers.MaxPoolLayerV1(pool_size=self.config['graphsim']['conv_pool_size'][i],
                stride=self.config['graphsim']['conv_pool_size'][i], num_similarity_matrices=self.num_gcn_layers))
            in_channels = self.config['graphsim']['conv_out_channels'][i]

        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(self.config['graphsim']['linear_size'][i],
                self.config['graphsim']['linear_size'][i+1]))

        self.scoring_layer = torch.nn.Linear(self.config['graphsim']['linear_size'][-1], 1)

    def GCN_pass(self, data):
        features, edge_index = data.x, data.edge_index
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)


        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def Conv_pass(self, similarity_matrices_list):
        features = [_.unsqueeze(1) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features);

            features = [torch.nn.functional.dropout(_,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)  for _ in features]
        return features

    def linear_pass(self, features):
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = torch.nn.functional.relu(features);
            features = torch.nn.functional.dropout(features,p=self.config['graphsim']['dropout'],
                                               training=self.training)
        return features

    def forward(self, batch_data,batch_data_sizes,batch_adj):

        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)

        query_abstract_features_list = self.GCN_pass(query_batch)
        query_abstract_features_list = [pad_sequence(torch.split(query_abstract_features_list[i], list(a), dim=0), batch_first=True) \
                                        for i in range(self.num_gcn_layers)]


        corpus_abstract_features_list = self.GCN_pass(corpus_batch)
        corpus_abstract_features_list = [pad_sequence(torch.split(corpus_abstract_features_list[i], list(b), dim=0), batch_first=True) \
                                          for i in range(self.num_gcn_layers)]

        similarity_matrices_list = [torch.matmul(query_abstract_features_list[i],\
                                    corpus_abstract_features_list[i].permute(0,2,1))
                                    for i in range(self.num_gcn_layers)]

        features = torch.cat(self.Conv_pass(similarity_matrices_list), dim=1).view(-1,
                              self.config['graphsim']['linear_size'][0])
        features = self.linear_pass(features);


        score_logits = self.scoring_layer(features)
        if self.av.is_sig:
          score = torch.sigmoid(score_logits)
          return score.view(-1)
        else:
          return score_logits.view(-1)

def dense_wasserstein_distance_v3(cost_matrix):
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(cost_matrix);

    return np.eye(cost_matrix.shape[0])[col_ind_lapjv];


class GOTSim(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GOTSim, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim

        #Conv layers
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.av.filters_1)
        self.conv2 = pyg_nn.GCNConv(self.av.filters_1, self.av.filters_2)
        self.conv3 = pyg_nn.GCNConv(self.av.filters_2, self.av.filters_3)
        self.num_gcn_layers = 3

        self.n1 = self.av.MAX_QUERY_SUBGRAPH_SIZE
        self.n2 = self.av.MAX_CORPUS_SUBGRAPH_SIZE
        self.insertion_constant_matrix = cudavar(self.av,99999 * (torch.ones(self.n1, self.n1)
                                                - torch.diag(torch.ones(self.n1))))
        self.deletion_constant_matrix = cudavar(self.av,99999 * (torch.ones(self.n2, self.n2)
                                                - torch.diag(torch.ones(self.n2))))


        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)

        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_1)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_2)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_3)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_1)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_2)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_3)))

    def GNN (self, data):
        """
        """
        gcn_feature_list = []
        features = self.conv1(data.x,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        gcn_feature_list.append(features)
        return gcn_feature_list


    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        batch_sz = len(batch_data)
        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)
        query_gcn_feature_list = self.GNN(query_batch)
        corpus_gcn_feature_list = self.GNN(corpus_batch)

        pad_main_similarity_matrices_list=[]
        pad_deletion_similarity_matrices_list = []
        pad_insertion_similarity_matrices_list = []
        pad_dummy_similarity_matrices_list = []
        for i in range(self.num_gcn_layers):

            q = pad_sequence(torch.split(query_gcn_feature_list[i], list(a), dim=0), batch_first=True)
            c = pad_sequence(torch.split(corpus_gcn_feature_list[i],list(b), dim=0), batch_first=True)
            q = F.pad(q,pad=(0,0,0,self.n1-q.shape[1],0,0))
            c = F.pad(c,pad=(0,0,0,self.n2-c.shape[1],0,0))
            #NOTE THE -VE HERE. BECAUSE THIS IS ACTUALLY COST MAT
            pad_main_similarity_matrices_list.append(-torch.matmul(q,c.permute(0,2,1)))

            pad_deletion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(q, self.deletion_params[i]))+\
                                                    self.insertion_constant_matrix)

            pad_insertion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(c, self.insertion_params[i]))+\
                                                     self.deletion_constant_matrix)

            pad_dummy_similarity_matrices_list.append(cudavar(self.av,torch.zeros(batch_sz,self.n2, self.n1, \
                                                      dtype=q.dtype)))


        sim_mat_all = []
        for j in range(batch_sz):
            for i in range(self.num_gcn_layers):
                a = pad_main_similarity_matrices_list[i][j]
                b =pad_deletion_similarity_matrices_list[i][j]
                c = pad_insertion_similarity_matrices_list[i][j]
                d = pad_dummy_similarity_matrices_list[i][j]
                s1 = qgraph_sizes[j]
                s2 = cgraph_sizes[j]
                sim_mat_all.append(torch.cat((torch.cat((a[:s1,:s2], b[:s1,:s1]), dim=1),\
                               torch.cat((c[:s2,:s2], d[:s2,:s1]), dim=1)), dim=0))


        sim_mat_all_cpu = [x.detach().cpu().numpy() for x in sim_mat_all]
        plans = [dense_wasserstein_distance_v3(x) for x in sim_mat_all_cpu ]
        mcost = [torch.sum(torch.mul(x,cudavar(self.av,torch.Tensor(y)))) for (x,y) in zip(sim_mat_all,plans)]
        sz_sum = qgraph_sizes.repeat_interleave(3)+cgraph_sizes.repeat_interleave(3)
        mcost_norm = 2*torch.div(torch.stack(mcost),sz_sum)
        scores_new =  self.ot_scoring_layer(mcost_norm.view(-1,3)).squeeze()
        #return scores_new.view(-1)

        if self.av.is_sig:
            return torch.sigmoid(scores_new).view(-1)
        else:
            return scores_new.view(-1)


class NeuroMatch(nm.OrderEmbedder):
    def __init__(self, input_dim, hidden_dim, av):
        super(NeuroMatch, self).__init__(input_dim, hidden_dim, av)

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        q_graphs,c_graphs = zip(*batch_data)
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)

        query_abstract_features = self.emb_model(query_batch)
        corpus_abstract_features = self.emb_model(corpus_batch)

        return self.predict((query_abstract_features, corpus_abstract_features))

def evaluate_embeddings_similarity(av,model,sampler):
  model.eval()
  d_pos = sampler.list_pos
  d_neg = sampler.list_neg
  d = d_pos + d_neg
  npos = len(d_pos)
  nneg = len(d_neg)

  pred = []

  n_batches = sampler.create_batches(d)
  for i in range(n_batches):
    #ignoring target values here since not needed for AP ranking score 
    batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
    pred.append( model(batch_data,batch_data_sizes,batch_adj).data)

  all_pred = torch.cat(pred,dim=0) 
  labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
  ap_score   = average_precision_score(labels.cpu(), all_pred.cpu())  

  q_graphs = list(range(len(sampler.query_graphs)))    
    
  all_ap = []

  for q_id in q_graphs:
    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
    npos = len(dpos)
    nneg = len(dneg)
    d = dpos+dneg
    if npos>0 and nneg>0:    
      n_batches = sampler.create_batches(d) 
      pred = []  
      for i in range(n_batches):
        batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
        pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
      all_pred = torch.cat(pred,dim=0) 
      labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
      ap   = average_precision_score(labels.cpu(), all_pred.cpu()) 
      all_ap.append(ap)  
  return ap_score, np.mean(all_ap)

def eval_edge_alignment(av,bestM, sampler):
    """
      bestM: usually best validation model is to be passed here for diagnostics
      sampler: evaluation data to be passed here. 
    """
    s = time.time()
    bestM.diagnostic_mode = True
    all_ap1 = []
    all_ap3 = []
    all_pos_hs1 = []
    all_neg_hs1 = []
    all_pos_hs3 = []
    all_neg_hs3 = []
    maskDiag = torch.ones(av.MAX_EDGES,av.MAX_EDGES).fill_diagonal_(0)
    q_graphs = list(range(len(sampler.query_graphs)))    
    for q_id in q_graphs:
        dpos = list(filter(lambda x:x[0][0]==q_id,sampler.list_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,sampler.list_neg))
        npos = len(dpos)
        nneg = len(dneg)
        d = dpos+dneg
        #if npos>0 and nneg>0:
        n_batches = sampler.create_batches(d)
        plans = []  
        qesim = []
        cesim = []
        qnormvals = [] 
        cnormvals = []
        for i in range(n_batches):
            batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
            plans.append( bestM(batch_data,batch_data_sizes,batch_adj))
            node_features, edge_features, from_idx, to_idx, graph_idx = bestM.get_graph(batch_data)
            edge_counts  = bestM.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
            from_idx_split = torch.split(from_idx, edge_counts, dim=0)
            to_idx_split = torch.split(to_idx, edge_counts, dim=0)
            batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
            prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k])for k in range(len(batch_data_sizes_flat))]
            to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
            from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]
            incidence_mat = [torch.zeros(av.MAX_EDGES,x) for x in batch_data_sizes_flat]
            from_idx_enum = [(torch.arange(a),b) for (a,b) in zip(edge_counts,from_idx_split_relabeled)]
            to_idx_enum = [(torch.arange(a),b) for (a,b) in zip(edge_counts,to_idx_split_relabeled)]
            for a,b in zip(incidence_mat,from_idx_enum):
                a[b] = 1
            for a,b in zip(incidence_mat,to_idx_enum):
                a[b] = 1
            edge_sim_mat = [torch.mul(maskDiag,(x@x.T)) for x in incidence_mat]
            qesim.append(cudavar(av,torch.stack(edge_sim_mat[0::2])))
            cesim.append(cudavar(av,torch.stack(edge_sim_mat[1::2]))) 
            normvals = [torch.sum(x) for x in incidence_mat]
            qnormvals.append(cudavar(av,torch.stack(normvals[0::2])))
            cnormvals.append(cudavar(av,torch.stack(normvals[1::2])))

        all_plans = torch.cat(plans,dim=0) 
        all_qesim  = torch.cat(qesim)
        all_cesim  = torch.cat(cesim) 
        all_qnormvals  =torch.cat(qnormvals)
        all_cnormvals  =torch.cat(cnormvals)
        all_hard_plans = []
        for p in all_plans:
            soft_plan = p.detach().cpu()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-soft_plan)
            all_hard_plans.append(torch.eye(av.MAX_EDGES)[col_ind])
        all_hard_plans = cudavar(av,torch.stack(all_hard_plans))


        hingeScore1 = -torch.sum(torch.nn.ReLU()(all_qesim - all_hard_plans@all_cesim@all_hard_plans.permute(0,2,1)),\
                      dim=(1,2))
        hingeScore3 = torch.div(hingeScore1,all_qnormvals)/2
        labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
        all_ap1.append(average_precision_score(labels.cpu(), hingeScore1.cpu()))
        all_ap3.append(average_precision_score(labels.cpu(), hingeScore3.cpu()))
        all_pos_hs1= all_pos_hs1 + hingeScore1[labels>0.5].tolist()
        all_neg_hs1= all_neg_hs1 + hingeScore1[labels<0.5].tolist()
        all_pos_hs3= all_pos_hs3 + hingeScore3[labels>0.5].tolist()
        all_neg_hs3= all_neg_hs3 + hingeScore3[labels<0.5].tolist()
    
    # Reverting back to original state
    bestM.diagnostic_mode = False
    return all_ap1, all_ap3 , all_pos_hs1, all_neg_hs1,all_pos_hs3, all_neg_hs3

def eval_node_alignment(av,bestM, sampler):
    """
      bestM: usually best validation model is to be passed here for diagnostics
      sampler: evaluation data to be passed here. 
    """
    bestM.diagnostic_mode = True
    all_ap1 = []
    all_ap2 = []
    all_ap3 = []
    all_pos_hs1 = []
    all_neg_hs1 = []
    all_pos_hs2 = []
    all_neg_hs2 = []
    all_pos_hs3 = []
    all_neg_hs3 = []
    q_graphs = list(range(len(sampler.query_graphs)))    
    for q_id in q_graphs:
        dpos = list(filter(lambda x:x[0][0]==q_id,sampler.list_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,sampler.list_neg))
        npos = len(dpos)
        nneg = len(dneg)
        d = dpos+dneg
        if npos>0 and nneg>0:
          n_batches = sampler.create_batches(d)
          plans = []  
          aq = []
          ac = []
          qsz = [] 
          csz = []
          for i in range(n_batches):
            batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(i)
            plans.append( bestM(batch_data,batch_data_sizes,batch_adj))
            a, b = zip(*batch_adj)
            aq.append(torch.stack(a))
            ac.append(torch.stack(b))
            a,b = zip(*batch_data_sizes)
            qsz.append(cudavar(av,torch.tensor(a)))
            csz.append(cudavar(av,torch.tensor(b)))
          all_plans = torch.cat(plans,dim=0) 
          all_aq  = torch.cat(aq)
          all_ac  = torch.cat(ac)  
          all_qsz = torch.cat(qsz)
          all_csz = torch.cat(csz)
          all_hard_plans = []
          for p in all_plans:
            soft_plan = p.detach().cpu()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-soft_plan)
            all_hard_plans.append(torch.eye(15)[col_ind])
          all_hard_plans = cudavar(av,torch.stack(all_hard_plans))
          all_qedge_sz = torch.sum(all_aq,dim=(1,2))/2
          all_cedge_sz = torch.sum(all_ac,dim=(1,2))/2

        hingeScore1 = -torch.sum(torch.nn.ReLU()(all_aq - all_hard_plans@all_ac@all_hard_plans.permute(0,2,1)),\
                      dim=(1,2))
        hingeScore2 = torch.div(hingeScore1,torch.mul(all_qsz,all_csz))
        hingeScore3 = torch.div(hingeScore1,all_qedge_sz)/2
        labels = cudavar(av,torch.cat((torch.ones(npos),torch.zeros(nneg))))
        all_ap1.append(average_precision_score(labels.cpu(), hingeScore1.cpu()))
        all_ap2.append(average_precision_score(labels.cpu(), hingeScore2.cpu()))
        all_ap3.append(average_precision_score(labels.cpu(), hingeScore3.cpu()))
        all_pos_hs1= all_pos_hs1 + hingeScore1[labels>0.5].tolist()
        all_neg_hs1= all_neg_hs1 + hingeScore1[labels<0.5].tolist()
        all_pos_hs2= all_pos_hs2 + hingeScore2[labels>0.5].tolist()
        all_neg_hs2= all_neg_hs2 + hingeScore2[labels<0.5].tolist()
        all_pos_hs3= all_pos_hs3 + hingeScore3[labels>0.5].tolist()
        all_neg_hs3= all_neg_hs3 + hingeScore3[labels<0.5].tolist()

    # Reverting back to original state     
    bestM.diagnostic_mode = False
    return all_ap1, all_ap2 , all_ap3 , all_pos_hs1, all_neg_hs1,\
           all_pos_hs2 , all_neg_hs2, all_pos_hs3, all_neg_hs3

def pairwise_ranking_loss_similarity(predPos, predNeg, margin):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)

def train(av,config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  val_data = OurMatchingModelSubgraphIsoData(av,mode="val")

  if av.TASK.startswith("matching_iso_var_18_gmn_sinkhorn_param_big_hinge_score_on_embeds"):
    logger.info("Loading model OurMatchingModelVar18_GMN_encoding_SinkhornParamBig_HingeScoreOnEmbeds")  
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c)")  
    model = Node_align_Node_loss(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_19_gmn_all"):
    logger.info("Loading model OurMatchingModelVar19_GMN_all")  
    logger.info("This uses GMN embedding model.No regularizer. ")  
    model = GMN_embed(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_27_gmn_edge_perm_sinkhorn_param_big_hinge_score_on_edges"):
    logger.info("Loading model OurMatchingModelVar27_GMN_encoding_EdgePerm_SinkhornParamBig_HingeScoreOnEdges")  
    logger.info("This uses GMN encoder to obtain bad of node embeddings per graph, which is then used to obtain bag of edge embeddings per graph - then parameterized sinkhorn with LRL and similarity computation using hinge scoring (H_q, PH_c) where H consists of edge embeds") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = ISONET(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_29_gmn_sinkhorn_param_big_hinge_score_on_edge_similarity_ff_adj_mask"):
    logger.info("Loading model OurMatchingModelVar29_GMN_encoding_SinkhornParamBig_HingeScoreOnEdgeSimilarityByFF_AdjMask")  
    logger.info("This uses GMN encoder followed by parameterized sinkhorn with LRL and similarity computation using hinge scoring (A_q, PA_cP^T) . Each entry of A is a single score forward and backward message for the edge obtained using a FF. On top of the similarity matrix, we put a mask corr to adjacency matrix")  
    model = Node_align_Edge_loss(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_graphsim"):
    logger.info("Loading model GraphSim")  
    logger.info("This is GraphSim model")  
    model = GraphSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("ir_modified_gotsim"):
    logger.info("Loading IR_modified_GotSim")  
    logger.info("This uses IR_modified_GotSim  model. ")  
    model = GOTSim(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("simgnn_noperm") :
    logger.info("Loading model SimGNN")
    logger.info("This loads the entire SimGNN model. Input feature is [1]. No node permutation is done after nx graph loading")
    model = SimGNN(av,1).to(device)
    train_data.data_type = "pyg"
    val_data.data_type = "pyg"
  elif av.TASK.startswith("gmn_match"):
    logger.info("Loading GMN_match")  
    logger.info("This uses GMN matching model.No regularizer. ")  
    model = GMN_match(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_34_gmn_embed_hinge"):
    logger.info("Loading GMN_embed_hinge")  
    logger.info("This uses GMN embedding model with hinge loss.No regularizer. ")  
    model = GMN_embed_hinge(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_35_gmn_match_hinge"):
    logger.info("Loading GMN_match_hinge")  
    logger.info("This uses GMN matching model with hinge loss.No regularizer. ")  
    model = GMN_match_hinge(av,config,1).to(device)
    logger.info(model)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_var_36_gmn_edge_perm_sinkhorn_param_big_sqeuc_score_on_edges"):
    logger.info("Loading model OurMatchingModelVar36_GMN_encoding_EdgePerm_SinkhornParamBig_SqEucScoreOnEdges")  
    logger.info("This uses GMN encoder to obtain bad of node embeddings per graph, which is then used to obtain bag of edge embeddings per graph - then parameterized sinkhorn with LRL and similarity computation using squared euclidean scoring (H_q, PH_c) where H consists of edge embeds") 
    #One more hack. 
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
    model = ISONET_Sym(av,config,1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"
  elif av.TASK.startswith("matching_iso_neuromatch"): 
    logger.info("Loading model neuromatch")   
    logger.info("This is neuromatch model")   
    model = NeuroMatch(1,av.neuromatch_hidden_dim,av).to(device) 
    logger.info(model) 
    train_data.data_type = "pyg" 
    val_data.data_type = "pyg" 
  else:
    logger.info("ALERT!! CHECK FOR ERROR")  

  save_initial_model(av,model)
  optimizer = torch.optim.Adam(model.parameters(),
                                    lr=av.LEARNING_RATE,
                                    weight_decay=av.WEIGHT_DECAY)  
  cnt =0
  for param in model.parameters():
        cnt=cnt+torch.numel(param)
  logger.info("no. of params in model: %s",cnt)
  es = EarlyStoppingModule(av,av.ES)

  run = 0
  while av.RUN_TILL_ES or run<av.NUM_RUNS:
    model.train()
    start_time = time.time()
    n_batches = train_data.create_stratified_batches()
    epoch_loss = 0
    start_time = time.time()
    for i in range(n_batches):
      batch_data,batch_data_sizes,target,batch_adj = train_data.fetch_batched_data_by_id(i)
      optimizer.zero_grad()
      prediction = model(batch_data,batch_data_sizes,batch_adj)
      #Pairwise ranking loss
      predPos = prediction[target>0.5]
      predNeg = prediction[target<0.5]
      losses = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), av.MARGIN)
      #losses = torch.nn.functional.mse_loss(target, prediction,reduction="sum")
      losses.backward()
      optimizer.step()
      epoch_loss = epoch_loss + losses.item()

    logger.info("Run: %d train loss: %f Time: %.2f",run,epoch_loss,time.time()-start_time)
    #TODO:SAVE model - If needed
    #if run%10==0:
    #save_model_at_epoch(av,model, run)
    start_time = time.time()
    ap_score,map_score = evaluate_embeddings_similarity(av,model,val_data)
    logger.info("Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f",run,ap_score,map_score, time.time()-start_time)
    if av.RUN_TILL_ES:
      if es.check([map_score],model,run):
        break
    run+=1
 
if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",                      type=bool,  default=True)
  ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--is_sig",                         type=bool,  default=False)
  ap.add_argument("--ES",                             type=int,   default=50)
  ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
  ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
  ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
  ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
  ap.add_argument("--MAX_GRAPH_SIZE",                 type=int,   default=0)
  ap.add_argument("--n_layers",                       type=int,   default=3)
  ap.add_argument("--conv_type",                      type=str,   default='SAGE')
  ap.add_argument("--method_type",                    type=str,   default='order')
  ap.add_argument("--skip",                           type=str,   default='learnable')
  ap.add_argument("--neuromatch_hidden_dim",          type=int,   default=10)
  ap.add_argument("--post_mp_dim",                    type=int,   default=64)
  ap.add_argument("--filters_1",                      type=int,   default=128)
  ap.add_argument("--filters_2",                      type=int,   default=64)
  ap.add_argument("--filters_3",                      type=int,   default=32)
  ap.add_argument("--dropout",                        type=float, default=0)
  ap.add_argument("--tensor_neurons",                 type=int,   default=16)
  ap.add_argument("--transform_dim" ,                 type=int,   default=16)
  ap.add_argument("--bottle_neck_neurons",            type=int,   default=16)
  ap.add_argument("--bins",                           type=int,   default=16)
  ap.add_argument("--histogram",                      type=bool,  default=False)
  ap.add_argument("--GMN_NPROPLAYERS",                type=int,   default=5)
  ap.add_argument("--MARGIN",                         type=float, default=0.1)
  ap.add_argument("--NOISE_FACTOR",                   type=float, default=1.0)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=2)
  ap.add_argument("--BATCH_SIZE",                     type=int,   default=128)
  ap.add_argument("--LEARNING_RATE",                  type=float, default=0.001)
  ap.add_argument("--WEIGHT_DECAY",                   type=float, default=5*10**-4)
  ap.add_argument("--FEAT_TYPE",                      type=str,   default="Onehot1",help="One/Onehot/Onehot1/Adjrow/Adjrow1/AdjOnehot")
  ap.add_argument("--CONV",                           type=str,   default="GCN",help="GCN/GAT/GIN/SAGE")
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag", help="TODO")
  ap.add_argument("--TASK",                           type=str,   default="OurMatchingSimilarity",help="TODO")

  av = ap.parse_args()

  if av.FEAT_TYPE == "Adjrow" or  av.FEAT_TYPE == "Adjrow1" or av.FEAT_TYPE == "AdjOnehot": 
      av.TASK = av.TASK + "_" + av.FEAT_TYPE
  if av.CONV != "GCN": 
      av.TASK = av.TASK + "_" + av.CONV
  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+str(time.time())
  set_log(av)
  logger.info("Command line")
  logger.info('\n'.join(sys.argv[:]))

  # Print configure
  config = get_default_config()
  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]
  config['graph_matching_net'] ['node_state_dim'] = 10
  #config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
  config['graph_matching_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['node_state_dim'] = 10
  #config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  
  #logger.info("av gmn_prop_param")
  #logger.info(av.GMN_NPROPLAYERS) 
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
  config['graphsim'] = {} 
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout 

  for (k, v) in config.items():
      logger.info("%s= %s" % (k, v))  

  # Set random seeds
  seed = config['seed']
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False
#  torch.backends.cudnn.benchmark = True


  av.dataset = av.DATASET_NAME
  train(av,config)


