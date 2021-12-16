import os
from os import path

import json
import random
from datetime import datetime, timedelta, time

import numpy as np
import torch
from torch import nn
import dgl

from tqdm.auto import tqdm

from models import GeometricGraphSim
from utils import tab_printer, calculate_sigmoid_loss
from utils import Metric

def process_pair_v2(data, global_labels):
    """
    :param path: graph pair data.
    :return data: Dictionary with data, also containing processed DGL graphs.
    """
    # print('Using v2 process_pair')
    edges_1 = data["graph_1"] #diff from v1
    edges_2 = data["graph_2"] #diff from v1

    edges_1 = np.array(edges_1, dtype=np.int64);
    edges_2 = np.array(edges_2, dtype=np.int64);
    G_1 = dgl.DGLGraph((edges_1[:,0], edges_1[:,1]));
    G_2 = dgl.DGLGraph((edges_2[:,0], edges_2[:,1]));
    G_1.add_edges(G_1.nodes(), G_1.nodes()) #diff from v1
    G_2.add_edges(G_2.nodes(), G_2.nodes()) #diff from v1
    
    edges_1 = torch.from_numpy(edges_1.T).type(torch.long)
    edges_2 = torch.from_numpy(edges_2.T).type(torch.long)
    data["edge_index_1"] = edges_1
    data["edge_index_2"] = edges_2
    
    features_1, features_2 = [], []

    for n in data["labels_1"]:
        features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

    for n in data["labels_2"]:
        features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        
    G_1.ndata['features'] = torch.FloatTensor(np.array(features_1));
    G_2.ndata['features'] = torch.FloatTensor(np.array(features_2));
    
    G_1.ndata['type'] = np.array(data["labels_1"]);
    G_2.ndata['type'] = np.array(data["labels_2"]);
    
    data['G_1'] = G_1;
    data['G_2'] = G_2;
    
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
    
    return data

def process_pair(data, global_labels):
    """
    :param path: graph pair data.
    :return data: Dictionary with data, also containing processed DGL graphs.
    """
    edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

    edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

    edges_1 = np.array(edges_1, dtype=np.int64);
    edges_2 = np.array(edges_2, dtype=np.int64);
    G_1 = dgl.DGLGraph((edges_1[:,0], edges_1[:,1]));
    G_2 = dgl.DGLGraph((edges_2[:,0], edges_2[:,1]));
    
    
    edges_1 = torch.from_numpy(edges_1.T).type(torch.long)
    edges_2 = torch.from_numpy(edges_2.T).type(torch.long)
    data["edge_index_1"] = edges_1
    data["edge_index_2"] = edges_2
    
    features_1, features_2 = [], []

    for n in data["labels_1"]:
        features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

    for n in data["labels_2"]:
        features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        
    G_1.ndata['features'] = torch.FloatTensor(np.array(features_1));
    G_2.ndata['features'] = torch.FloatTensor(np.array(features_2));
    
    G_1.ndata['type'] = np.array(data["labels_1"]);
    G_2.ndata['type'] = np.array(data["labels_2"]);
    
    data['G_1'] = G_1;
    data['G_2'] = G_2;
    
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
    
    return data

def process_pair_with_minmax_scaling(data, global_labels, min_ged, max_ged, scale_by_graph_size=False):
    """
    :param path: graph pair data.
    :return data: Dictionary with data, also containing processed DGL graphs.
    """
    edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

    edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

    edges_1 = np.array(edges_1, dtype=np.int64);
    edges_2 = np.array(edges_2, dtype=np.int64);
    G_1 = dgl.DGLGraph((edges_1[:,0], edges_1[:,1]));
    G_2 = dgl.DGLGraph((edges_2[:,0], edges_2[:,1]));
    
    features_1, features_2 = [], []

    for n in data["labels_1"]:
        features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

    for n in data["labels_2"]:
        features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        
        
    G_1.ndata['features'] = torch.FloatTensor(np.array(features_1));
    G_2.ndata['features'] = torch.FloatTensor(np.array(features_2));
    
    data['G_1'] = G_1;
    data['G_2'] = G_2;
    
    #data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()

    if scale_by_graph_size:
        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    else:
        norm_ged = data["ged"]
    norm_ged = torch.from_numpy(np.array([(norm_ged - min_ged) / (max_ged - min_ged)]))
    data["target"] = norm_ged
    
    return data

# Weight initialization
def weights_init_v1(m):
    """weight initialization"""
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2dSameLayer') != -1:
        print(classname)
        for p in m.parameters():
            nn.init.normal_(p.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        print(classname)
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.normal_(m.weight.data, 0.0, 1)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)
            
def weights_init_v2(m):
    """weight initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv2dSameLayer') != -1:
        for p in m.parameters():
            nn.init.xavier_uniform_(p.data)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None and m.bias.data is not None:
            nn.init.constant(m.bias.data, 0)            

class Trainer(object):
    """
    Model trainer.
    """
    def __init__(self, args, training_pairs, validation_pairs, testing_pairs, device, model_class=GeometricGraphSim):
        """
        :param args: Arguments object.
        """
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.args = args
        self.model_class = model_class
        
        self.training_pairs = training_pairs
        self.validation_pairs = validation_pairs
        self.testing_pairs = testing_pairs
        self.initial_label_enumeration()

        
        # Tracking training stats 
        self.epoch = 0
        self.best_val_mse = None
        self.val_mses = []
        self.best_val_metric = None
        self.val_metrics = []
        self.losses = []
        self.early_stop = False
        self.counter = 0
        self.epoch_times = []

        
        self.device = device
        self.setup_model()
        self.initialize_model()
        
    def setup_model(self):
        """
        Creating a GraphSim.
        """
        self.model = self.model_class(self.args, self.number_of_labels)
        print(self.model)
        
        self.model = self.model.to(self.device)
        if 'init_weight_method' in self.args and self.args.init_weight_method:
            print('Init weight method: {}'.format(self.args.init_weight_method))
            if self.args.init_weight_method == 'v1':
                self.model.apply(weights_init_v1)
            elif self.args.init_weight_method == 'v2':
                self.model.apply(weights_init_v2)
        self.best_model = self.model_class(self.args, self.number_of_labels)
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                          lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
        
    def initialize_model(self):
        if path.exists(self.args.exp_dir):
            checkpoint_files = sorted([d for d in os.listdir(self.args.exp_dir) if 'checkpoint_' in d])
            if len(checkpoint_files) > 0: #there exist some model, load them
                checkpoint_path = path.join(self.args.exp_dir, checkpoint_files[-1])
                if self.args.verbose >= 1:
                    print('Loading existing checkpoint: {}'.format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
                self.best_val_mse = checkpoint['best_val_mse']
                self.val_mses = checkpoint['val_mses']
                if 'best_val_metric' in checkpoint:
                    self.best_val_metric = checkpoint['best_val_metric']
                    self.val_metrics = checkpoint['val_metrics']
                self.losses = checkpoint['losses']
                self.early_stop = checkpoint['early_stop']
                self.epoch = checkpoint['epoch'] + 1
                self.counter = checkpoint['counter']
                self.epoch_times = checkpoint['epoch_times']
                if self.args.verbose >= 2:
                    print('Starting from epoch {}'.format(self.epoch))
                    
    def save_checkpoint(self, is_best_model=False):
        """Saves model"""
        if not is_best_model:
            save_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(self.epoch))
            if self.args.verbose >= 3:
                print('Save checkpoint: {}'.format(save_path))
            torch.save({
                'epoch': self.epoch,
                'counter': self.counter,
                'best_val_mse': self.best_val_mse,
                'val_mses': self.val_mses, 
                'best_val_metric': self.best_val_metric,
                'val_metrics': self.val_metrics, 
                'losses': self.losses,
                'early_stop': self.early_stop,
                'model': self.model.state_dict(),
                'epoch_times': self.epoch_times,
                'optimizer': self.optimizer.state_dict()
            }, save_path)
            
        else: 
            save_path = path.join(self.args.exp_dir, 'best_checkpoint.pt')
            if self.args.verbose >= 3:
                print('Save best checkpoint: {}'.format(save_path))
            torch.save({
                'epoch': self.epoch,
                'best_val_mse': self.best_val_mse,
                'best_val_metric': self.best_val_metric,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, save_path)
        
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        graph_pairs = self.training_pairs + self.testing_pairs + self.validation_pairs
        self.global_labels = set()
        for data in graph_pairs:
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
        
        if 'ged_scaling' in self.args:
            print('Scale GED by method: {}'.format(self.args.ged_scaling))
            if self.args.ged_scaling == 'minmax_scaling_v1':
                geds = [pair['ged'] /(0.5*(len(pair["labels_1"])+len(pair["labels_2"]))) for pair in graph_pairs]
                self.min_ged, self.max_ged = np.min(geds), np.max(geds)
                
                self.training_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=True) for graph_pair in tqdm(self.training_pairs)]
                self.testing_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=True) for graph_pair in tqdm(self.testing_pairs)]
                self.validation_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=True) for graph_pair in tqdm(self.validation_pairs)]
            elif self.args.ged_scaling == 'minmax_scaling_v2':
                geds = [pair['ged'] for pair in graph_pairs]
                self.min_ged, self.max_ged = np.min(geds), np.max(geds)
                
                self.training_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=False) for graph_pair in tqdm(self.training_pairs)]
                self.testing_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=False) for graph_pair in tqdm(self.testing_pairs)]
                self.validation_pairs_dgl = [process_pair_with_minmax_scaling(graph_pair, self.global_labels, self.min_ged, self.max_ged, scale_by_graph_size=False) for graph_pair in tqdm(self.validation_pairs)]
            
        else:
            print('Scale GED by original method (exponential)')
            self.training_pairs_dgl = [process_pair(graph_pair, self.global_labels) for graph_pair in tqdm(self.training_pairs)]
            self.testing_pairs_dgl = [process_pair(graph_pair, self.global_labels) for graph_pair in tqdm(self.testing_pairs)]
            self.validation_pairs_dgl = [process_pair(graph_pair, self.global_labels) for graph_pair in tqdm(self.validation_pairs)]

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_pairs_dgl)
        batches = []
        for graph in range(0, len(self.training_pairs_dgl), self.args.batch_size):
            batches.append(self.training_pairs_dgl[graph:graph+self.args.batch_size])
        return batches
    
    def collate(self, samples):
        # The input `samples` is a list of dicts
        G1_list = [_['G_1'] for _ in samples];
        G2_list = [_['G_2'] for _ in samples];
        features_1 = torch.stack([_['features_1'] for _ in samples])
        features_2 = torch.stack([_['features_2'] for _ in samples])
        labels = torch.tensor([_['target'] for _ in samples])
        
        batched_graph_G1 = dgl.batch(G1_list)
        batched_graph_G2 = dgl.batch(G2_list)
        return batched_graph_G1, batched_graph_G2, features_1, features_2, torch.tensor(labels);

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        G_1 = data["G_1"]
        G_2 = data["G_2"]

        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["G_1"] = G_1
        new_data["G_2"] = G_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))

        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        predictions, logits = self.model(batch);
        target = torch.tensor([data["target"] for data in batch]).to(self.device);
        
        losses = torch.nn.functional.mse_loss(target, predictions);
        losses.backward()
        self.optimizer.step()
        loss = losses.item()
        return loss       
    
    def track_state(self, val_mse, is_final_epoch=False):
        if self.best_val_mse is None: #initial epoch
            self.best_val_mse = val_mse
            self.val_mses.append(val_mse)
            self.save_checkpoint(is_best_model=True)
        else:
            min_mse = min(self.val_mses[-self.args.patience:])
            if val_mse > min_mse + self.args.delta: #not improving from last min
                self.val_mses.append(val_mse)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else: #improve from last min
                if self.args.verbose >= 1:
                    print(f'Validation MSE decreased ({min_mse:.6f} --> {val_mse:.6f}).')
                self.val_mses.append(val_mse)
                self.counter = 0

            if val_mse < self.best_val_mse: #If this is the best model
                if self.args.verbose >= 2:
                    print(f'Best MSE ({self.best_val_mse:.6f} --> {val_mse:.6f}).  Will save best model.')
                self.best_val_mse = val_mse
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()

    def track_metric_state(self, val_metric, is_final_epoch=False):
        if self.best_val_metric is None: #initial epoch
            self.best_val_metric = val_metric
            self.val_metrics.append(val_metric)
            self.save_checkpoint(is_best_model=True)
        else:
            min_metric = min(self.val_metrics[-self.args.patience:])
            if val_metric < min_metric - self.args.delta: #not improving from last min
                self.val_metrics.append(val_metric)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else: #improve from last min
                if self.args.verbose >= 1:
                    print(f'Validation Metric increased ({min_metric:.6f} --> {val_metric:.6f}).')
                self.val_metrics.append(val_metric)
                self.counter = 0

            if val_metric > self.best_val_metric: #If this is the best model
                if self.args.verbose >= 2:
                    print(f'Best Metric ({self.best_val_metric:.6f} --> {val_metric:.6f}).  Will save best model.')
                self.best_val_metric = val_metric
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()            
    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        
        self.model.train()
        epochs = tqdm(range(self.epoch, self.args.epochs), leave=True, desc="Epoch")
        
        iters_per_stat = self.args.iters_per_stat if 'iters_per_stat' in self.args else 5
        metric = Metric(self.validation_pairs);
        evaluation_frequency = self.args.evaluation_frequency if 'evaluation_frequency' in self.args else 1
        for self.epoch in epochs:
            if self.early_stop:
                print('Early stopping!')
                break

            epoch_start_time = datetime.now()
            
            self.model.train()
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0

            for index, batch in tqdm(enumerate(batches), total=len(batches), mininterval=2):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                if index % iters_per_stat == 0 or index == len(batch)-1:
                    print("[Epoch %04d][Iter %d/%d] (Loss=%.05f)" % (self.epoch, index, len(batches), round(loss, 5)))

            epoch_duration = (datetime.now() - epoch_start_time)
            self.epoch_times.append(epoch_duration.total_seconds())
            print('[Epoch {}]: Finish in {:.2f} sec ({:.2f} min).'.format(
                    self.epoch, epoch_duration.total_seconds(), epoch_duration.total_seconds() / 60))
            
            self.losses.append(loss)
            validation_mse = self.score(test_pairs=self.validation_pairs_dgl);
            baseline_mse = self.baseline_score(test_pairs=self.validation_pairs_dgl);
            print("Validation MSE: {:.05f}, Baseline: {:.05f}".format(validation_mse, baseline_mse));
            
            is_final_epoch=(self.epoch+1 == self.args.epochs)
            if 'early_stopping_metric' in self.args:
                if self.epoch % evaluation_frequency == 0 or is_final_epoch:
                    print(f'Evaluating using metric: {self.args.early_stopping_metric}')
                    eval_start_time = datetime.now()
                    validation_predictions = self.predict(self.validation_pairs);
                    valid_spearman = metric.spearman(validation_predictions, mode="macro", unnormalized=False)
                    valid_kendalltau = metric.kendalltau(validation_predictions, mode="macro", unnormalized=False)
                    valid_precision = metric.average_precision_at_k(validation_predictions, k=10, unnormalized=False)
                    valid_mae = metric.mae(validation_predictions, unnormalized=False)
                    valid_mse = metric.mse(validation_predictions, unnormalized=False)
                    if self.args.early_stopping_metric == 'spearman':
                        validation_metric = valid_spearman
                    elif self.args.early_stopping_metric == 'kendalltau':
                        validation_metric = valid_kendalltau
                    elif self.args.early_stopping_metric == 'precision':
                        validation_metric = valid_precision
                    elif self.args.early_stopping_metric == 'mae':
                        validation_metric = valid_mae
                    elif self.args.early_stopping_metric == 'mse':
                        validation_metric = valid_mse
                    else:
                        raise Exception(f'not supported metric: {self.args.early_stopping_metric}')
                    
                    self.track_metric_state(validation_metric, is_final_epoch=is_final_epoch)
                    print(f'Evaluation: {validation_metric:.05f} (finishes in {(datetime.now() - eval_start_time).total_seconds()})')
                epochs.set_postfix(
                    **{self.args.early_stopping_metric: '{:.05f}'.format(self.best_val_metric)}
                )
            else:
                # early stopping
                self.track_state(validation_mse, is_final_epoch=is_final_epoch)
                epochs.set_postfix(mse='{:.05f}'.format(self.best_val_mse))
        validation_mse = self.score(test_pairs=self.validation_pairs_dgl);
        print("Final Validation MSE: ", validation_mse);

    def score(self, test_pairs):
        """
        Scoring on the test set.
        
        
        """
        self.model.eval()
        
        prediction, logits = self.model(test_pairs)
        prediction = prediction.detach().cpu().numpy()
        scores = calculate_sigmoid_loss(prediction, test_pairs)
        
        return np.mean(scores)

    def load_best_model(self, load_path=None):
        if self.best_model is None:
            self.best_model = self.model.clone()
            
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'best_checkpoint.pt')
        if self.args.verbose >= 2:
            print('Load best model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.best_model.load_state_dict(checkpoint['model'])
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
    def load_model(self, epoch=None, load_path=None):
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(epoch))
        if self.args.verbose >= 2:
            print('Load model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.model = self.best_model.to(self.device)
        self.model.eval()
            
    def score_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        prediction, logits = self.best_model(test_pairs)
        prediction = prediction.detach().cpu().numpy()
        scores = calculate_sigmoid_loss(prediction, test_pairs)
        return np.mean(scores)
    
    def predict_best_model(self, test_pairs, load_path=None):
        """
        Scoring on the test set.
        """
        self.load_best_model(load_path=load_path)
        self.best_model.eval()

        prediction, logits = self.best_model(test_pairs)
        prediction = prediction.detach().cpu().numpy()

        return prediction
        
    def predict(self, test_pairs):
        """
        Scoring on the test set.
        """
        self.model.eval()
        prediction, logits = self.model(test_pairs)
        prediction = prediction.detach().cpu().numpy()
        
        return prediction
    
    def baseline_score(self, test_pairs):
        """
        Baeline Scoring on the test set.
        """
        self.model.eval()
        scores = []
        average_ged = np.mean([data["target"] for data in test_pairs]);
        base_error = np.mean([(data["target"]-average_ged)**2 for data in test_pairs])
        
        return base_error;
    
class ExampleTrainer(Trainer):
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for data in batch:
            target = data["target"].cuda()
            prediction, logits = self.model(data['G_1'], data['G_2'])
            losses = losses + torch.nn.functional.mse_loss(target, prediction)
        losses = losses/len(batch);
        losses.backward()
        self.optimizer.step()
        loss = losses.item()
        return loss
    
    def score(self, test_pairs):
        """
        Scoring on the test set.
        """
        self.model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, logits = self.model(data['G_1'], data['G_2'])
            prediction = prediction.detach().cpu().numpy()
            scores.append(calculate_sigmoid_loss(prediction, data))
        return np.mean(np.mean(scores))
    
    def predict(self, test_pairs):
        """
        Scoring on the test set.
        """
        self.model.eval()
        prediction = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, one_logits = self.model(data['G_1'], data['G_2'])
            prediction.append(one_prediction.detach().cpu().numpy())
        prediction = np.concatenate(prediction)
        
        return prediction
    
    def score_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, logits = self.best_model(data['G_1'], data['G_2'])
            prediction = prediction.detach().cpu().numpy()
            scores.append(calculate_sigmoid_loss(prediction, data))
        return np.mean(scores)
    
    
    def predict_best_model(self, test_pairs, load_path=None):
        """
        Scoring on the test set.
        """
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        prediction = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, one_logits = self.best_model(data['G_1'], data['G_2'])
            prediction.append(one_prediction.detach().cpu().numpy())
        prediction = np.concatenate(prediction)
        
        return prediction
    
class ExampleTrainerV2(ExampleTrainer):
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        graph_pairs = self.training_pairs + self.testing_pairs + self.validation_pairs
        self.global_labels = set()
        for data in graph_pairs:
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
        
        print('Scale GED by original method (exponential)')
        self.training_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels) for graph_pair in tqdm(self.training_pairs, mininterval=2)]
        self.testing_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels) for graph_pair in tqdm(self.testing_pairs, mininterval=2)]
        self.validation_pairs_dgl = [process_pair_v2(graph_pair, self.global_labels) for graph_pair in tqdm(self.validation_pairs, mininterval=2)] 
    
class GeometricExampleTrainer(ExampleTrainer):
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for data in batch:
            target = data["target"].cuda()
            prediction, logits = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(target, prediction)
        losses = losses/len(batch);
        losses.backward()
        self.optimizer.step()
        loss = losses.item()
        return loss
    
    def score(self, test_pairs):
        """
        Scoring on the test set.
        """
        self.model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, logits = self.model(data)
            prediction = prediction.detach().cpu().numpy()
            scores.append(calculate_sigmoid_loss(prediction, data))
        return np.mean(np.mean(scores))
    
    def predict(self, test_pairs):
        """
        Scoring on the test set.
        """
        self.model.eval()
        prediction = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, one_logits = self.model(data)
            prediction.append(one_prediction.detach().cpu().numpy())
        prediction = np.concatenate(prediction)
        
        return prediction
    
    def score_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, logits = self.best_model(data)
            prediction = prediction.detach().cpu().numpy()
            scores.append(calculate_sigmoid_loss(prediction, data))
        return np.mean(scores)
    
    
    def predict_best_model(self, test_pairs, load_path=None, tqdm=tqdm):
        """
        Scoring on the test set.
        """
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        prediction = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, one_logits = self.best_model(data)
            prediction.append(one_prediction.detach().cpu().numpy())
        prediction = np.concatenate(prediction)
        
        return prediction    