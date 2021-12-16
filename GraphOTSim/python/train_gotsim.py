import os
from os import path
from argparse import Namespace

#from utils import tab_printer, calculate_sigmoid_loss

from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, math

import json
import glob
import torch
import numpy as np

#from train import ExampleTrainerV2
#from utils import Metric, summarize_results
#from models import GOTSim

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run GraphSim.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="/data/graph_search/PTC_ged/",
                        help="Folder with graphs, should contain a subfolder for each fold.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs. Default is 5.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=1024,
                        help="Bumber of graph pairs in a mini-batch. Default is 32.")

    parser.add_argument("--interpolate_size",
                        type=int,
                        default=54,
                        help="Size of interpolation. Default is 54.")
    
    parser.add_argument("--interpolate_mode",
                        type=str,
                        default='bilinear',
                        help="Algorithm for interpolation: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default is 'bilinear'.")

    parser.add_argument("--gcn_size",
                        type=int,
                        default=[128,64,32],
                        nargs='+',
                        help="List of neurons for the convolution layers. Default is 128 64 32.")

    parser.add_argument("--conv_kernel_size",
                        type=int,
                        default=[25,10,4,2],
                        nargs='+',
                        help="List of kernel sizes for the 2D convolution layers. Default is 25 10 4 2.")

    parser.add_argument("--conv_out_channels",
                        type=int,
                        default=[16,32,64,128],
                        nargs='+',
                        help="List of num out channels for the 2D convolution layers. Default is 16 32 64 128.")

    parser.add_argument("--conv_pool_size",
                        type=int,
                        default=[3,3,3,2],
                        nargs='+',
                        help="List of pooling kernel size for the 2D convolution layers. Default is 3 3 3 2.")

    parser.add_argument("--linear_size",
                        type=int,
                        default=[384, 256, 128, 64, 32, 16, 8, 4],
                        nargs='+',
                        help="List of linear layer sizes for the final feedforward network. Default is 384 256 128 64 32 16 8 4.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--folds",
                        type=int,
                        default=5,
                        help="Number of folds. Default is 5.")

    parser.add_argument("--patience",
                        type=int,
                        default=1,
                        help="Patience for early stopping. Default is 1.")
    parser.add_argument("--use_pairnorm", action="store_true", default=False)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--exp_label", type=str)

    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--save_frequency", type=int, default=5)
    
    parser.add_argument("--distance_type", type=str, default='negative_dot')
    parser.add_argument("--early_stopping_metric", type=str, default='precision')
    parser.add_argument("--evaluation_frequency", type=int, default=1)
    parser.add_argument("--patient", type=int, default=3)
    return parser.parse_args()

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()

    args.exp_label = f"{args.use_pairnorm}_{args.distance_type}_{args.batch_size}_{args.learning_rate}"
    
    tab_printer(args)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    graph_pairs = [];

    test_results = [None for _ in range(args.folds)]
    for fold in range(args.folds):
        graph_location = glob.glob(os.path.join(args.dataset, str(fold), "*.json"));
        temp_graph_pairs = [];
        for one_path in graph_location:
            temp_graph_pairs.append(json.load(open(one_path)));
        graph_pairs.append(temp_graph_pairs);
            
    for fold in range(args.folds):
        args.exp_dir = path.join(args.basedir, args.exp_label, 'fold_{}'.format(fold))
        
        print('Setting random seed={}'.format(fold))
        torch.manual_seed(fold)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(fold)
        
        if path.exists(args.exp_dir):
            print('Experiment dir exists...')
        else:
            print('Creating experiment dir: {}'.format(args.exp_dir))
            os.makedirs(args.exp_dir)
        
        test_fold = fold%args.folds;
        valid_fold = (fold+1)%args.folds;
        train_folds = [i%args.folds for i in range(1, args.folds)];
        training_pairs = sum([graph_pairs[i] for i in train_folds], []);
        validation_pairs = graph_pairs[valid_fold];
        testing_pairs = graph_pairs[test_fold];
                       
        
        trainer = ExampleTrainerV2(args, training_pairs, validation_pairs, testing_pairs, device, 
                                   model_class=GOTSim)
        trainer.fit()
        
        #Metrics
        metric = Metric(trainer.testing_pairs);
        test_predictions = trainer.predict_best_model(trainer.testing_pairs);
        #mse
        test_mse = metric.mse(test_predictions, unnormalized=False)
        #mae
        test_mae = metric.mae(test_predictions, unnormalized=False)
        #p@10
        test_precision = metric.average_precision_at_k(test_predictions, k=10, unnormalized=False)
        #spearman
        test_spearman = metric.spearman(test_predictions, mode="macro", unnormalized=False)
        #kendalltau
        test_kendalltau = metric.kendalltau(test_predictions, mode="macro", unnormalized=False)
        
        test_results[fold] = Namespace(mse=test_mse, mae=test_mae, 
                                       precision=test_precision, spearman=test_spearman,
                                      kendalltau=test_kendalltau)
        print(f'[Fold Test {fold}]: mse {test_mse:.05f}, mae {test_mae:.05f}, precision {test_precision:.05f}, spearman {test_spearman:.05f}, kendalltau {test_kendalltau:.05f}')
        
    return test_results
    
if __name__ == "__main__":
    results = main()    
