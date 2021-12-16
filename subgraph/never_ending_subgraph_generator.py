import argparse
from subgraph.graphs import TUDatasetGraph
import pickle
from common import logger, set_log
from subgraph.sampler import OnTheFlySubgraphSampler
import time
import networkx as nx
import numpy as np
import os
import sys
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import itertools


def check_iso(data):
  gc,gq = data
  return nx.algorithms.isomorphism.GraphMatcher(gc, gq).subgraph_is_isomorphic()


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--MIN_SUBGRAPH_SIZE",              type=int,   default=5)
  ap.add_argument("--MAX_SUBGRAPH_SIZE",              type=int,   default=10)
  ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
  ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
  ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
  ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag")
  ap.add_argument("--TASK",                           type=str,   default="NESGIso",help="PermGnnPointEmbedBON/PermGnnPointEmbedBOE")

  av = ap.parse_args()

  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME
  set_log(av)
  logger.info("Command line")
  logger.info('\n'.join(sys.argv[:]))


  tu_graph = TUDatasetGraph(av)
  graphs = tu_graph.get_nx_graph()

  #tu_graph = TUDatasetGraph(av)
  #tu_nx_g = tu_graph.get_nx_graph()
  #graphs = []
  #for i in range(len(tu_nx_g)):
  #  g = tu_nx_g[i]
  #  if len(g)>10 and len(g)<25:
  #    graphs.append(g)
  
  no_of_query_subgraphs = 100
  no_of_corpus_subgraphs = 800

  fp = av.DIR_PATH + "/Datasets/preprocessed/" + av.DATASET_NAME +"80k_corpus_subgraphs_" + \
              str(no_of_corpus_subgraphs)+"_min_"+str(av.MIN_CORPUS_SUBGRAPH_SIZE) + "_max_" + \
              str(av.MAX_CORPUS_SUBGRAPH_SIZE)+".pkl"
  logger.info("Sampling corpus subgraphs")
  subgraph_list, anchor_list, subgraph_id_list = [], [], []
  av.MIN_SUBGRAPH_SIZE = av.MIN_CORPUS_SUBGRAPH_SIZE
  av.MAX_SUBGRAPH_SIZE = av.MAX_CORPUS_SUBGRAPH_SIZE
  subgraph_sampler = OnTheFlySubgraphSampler(av,graphs)
  for i in range(no_of_corpus_subgraphs):
    sgraph,anchor,sgraph_id = subgraph_sampler.sample_subgraph()
    subgraph_list.append(sgraph)
    anchor_list.append(anchor)
    subgraph_id_list.append(sgraph_id)
  corpus_subgraph_list,corpus_anchor_list, corpus_subgraph_id_list =subgraph_list,anchor_list,subgraph_id_list

  print(fp)
  with open(fp, 'wb') as f:
    pickle.dump((corpus_subgraph_list,corpus_anchor_list,corpus_subgraph_id_list),f)

  fp = av.DIR_PATH + "/Datasets/preprocessed/" + av.DATASET_NAME +"80k_query_subgraphs_" + \
              str(no_of_query_subgraphs)+"_min_"+str(av.MIN_QUERY_SUBGRAPH_SIZE) + "_max_" + \
              str(av.MAX_QUERY_SUBGRAPH_SIZE)+".pkl"
  logger.info("Sampling query subgraphs")

  av.MIN_SUBGRAPH_SIZE = av.MIN_QUERY_SUBGRAPH_SIZE
  av.MAX_SUBGRAPH_SIZE = av.MAX_QUERY_SUBGRAPH_SIZE
  subgraph_sampler = OnTheFlySubgraphSampler(av,graphs)
  query_subgraph_list,query_anchor_list, query_subgraph_id_list = [],[],[]
  sstart = time.time()
  rel_dict = {}
  n_queries =0
  
  with ProcessPool(max_workers=100) as pool:
    while n_queries <no_of_query_subgraphs:
      start = time.time()
  
      sgraph,anchor,sgraph_id = subgraph_sampler.sample_subgraph()
      pos_c,neg_c = [],[]
      
      future = pool.map(check_iso, zip(corpus_subgraph_list,itertools.repeat(sgraph)), timeout=30)
      iterator = future.result()
  
      for c_i in range(no_of_corpus_subgraphs): 
          try:
              result = next(iterator)
              #assert(q_i==result[0] and c_i==result[1])
              if result ==1:
                  pos_c.append(c_i)
              else:
                  neg_c.append(c_i)
              
          except StopIteration:
              break
          except TimeoutError as error:  
              #rels[c_i][q_i] = False
              neg_c.append(c_i)
              logger.info(str(c_i)+" " + "Timeout")            
           
      if len(neg_c) ==0:
          r = 0
      else:  
          r = len(pos_c)/len(neg_c)
      if r>=0.1 and r<=0.4:
          logger.info("q: %s ratio : %s", n_queries,r)
          query_subgraph_list.append(sgraph)
          query_anchor_list.append(anchor)
          query_subgraph_id_list.append(sgraph_id)
          rel_dict[n_queries] = {}
          rel_dict[n_queries]['pos'] = pos_c
          rel_dict[n_queries]['neg'] = neg_c
          n_queries = n_queries+1
      else: 
          logger.info("Discarded due to raito : %s",r)
      logger.info("time to decide: %s", time.time()-start)

  print(fp)
  with open(fp, 'wb') as f:
    pickle.dump((query_subgraph_list,query_anchor_list,query_subgraph_id_list),f)



  subgraph_rel_type = "nx_is_subgraph_iso"
  fp = av.DIR_PATH + "/Datasets/preprocessed/" + av.DATASET_NAME + "80k_query_" + \
              str(no_of_query_subgraphs)+ "_corpus_" + str(no_of_corpus_subgraphs) + \
              "_minq_"+str(av.MIN_QUERY_SUBGRAPH_SIZE) + "_maxq_" + \
              str(av.MAX_QUERY_SUBGRAPH_SIZE)+"_minc_"+str(av.MIN_CORPUS_SUBGRAPH_SIZE) + "_maxc_" + \
              str(av.MAX_CORPUS_SUBGRAPH_SIZE)+"_rel_" + subgraph_rel_type +".pkl"
  with open(fp, 'wb') as f:
    pickle.dump(rel_dict,f)
  
  logger.info("Total time: %s", time.time()-sstart)
