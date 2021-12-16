from common import logger
import os
import torch
import torch.nn as nn

def load_model(av):
  """
  """
  #TODO

def save_model(av,model,optimizerPerm, optimizerFunc, epoch, saveAllEpochs = True):
  """
  """
  #TODO


def pairwise_ranking_loss(predPos, predNeg, margin):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)
    #return sum_loss


def load_model_at_epoch(av,epoch):
  """
    :param av           : args
    :return checkpoint  : dict containing model state dicts and av  
  """
  load_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(load_dir):
    os.makedirs(load_dir)
    #raise Exception('{} does not exist'.format(load_dir))
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name
  load_prefix = os.path.join(load_dir, name)
  load_path = '{}_epoch_{}'.format(load_prefix, epoch)
  if os.path.exists(load_path):
    logger.info("loading model from %s",load_path)
    checkpoint = torch.load(load_path)
  else: 
    checkpoint= None
  return checkpoint


def save_model_at_epoch(av,model, epoch):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :param epoch         : epoch no. 
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name
  save_prefix = os.path.join(save_dir, name)
  save_path = '{}_epoch_{}'.format(save_prefix, epoch)

  logger.info("saving model to %s",save_path)
  output = open(save_path, mode="wb")
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'av' : av,
            }, output)
  output.close()

def save_initial_model(av,model):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, "initialModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name
  save_prefix = os.path.join(save_dir, name)
  #save_path = '{}_epoch_{}'.format(save_prefix, epoch)

  logger.info("saving initial model to %s",save_prefix)
  output = open(save_prefix, mode="wb")
  torch.save({
            'model_state_dict': model.state_dict(),
            'av' : av,
            }, output)
  output.close()


def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return x.cuda() if av.has_cuda and av.want_cuda else x

def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = cudavar(av,torch.rand(shape).float())
  return -torch.log(eps - torch.log(U + eps))



