import torch
import random
import numpy as np
from tqdm import tqdm_notebook as tqdm, trange
import dgl

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.gcn_size[-1],
                                                             self.args.gcn_size[-1]))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.gcn_size[-1],
                                                             self.args.gcn_size[-1],
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.gcn_size[-1]))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.gcn_size[-1], -1))
        scoring = scoring.view(self.args.gcn_size[-1], self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores    

class Conv2dSameLayer(torch.nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, num_sim_matrices):
        super(Conv2dSameLayer, self).__init__()
        
        self.num_sim_matrices = num_sim_matrices
        
        bias = True
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        
        self.ops = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.ReflectionPad2d((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)) for i in range(self.num_sim_matrices)])
        
    def forward(self, features):
        results = []
        
        for i in range(self.num_sim_matrices):
            results.append(self.ops[i](features[:,i,:,:,:]).unsqueeze(1))
        
        return torch.tanh(torch.cat(results, dim=1))

class MaxPool2dLayer(torch.nn.Module):
    def __init__(self, kernel_size, stride, num_sim_matrices):
        super(MaxPool2dLayer, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_sim_matrices = num_sim_matrices
        
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        
        #self.ops = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride) for i in range(self.num_sim_matrices)])
        self.ops = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.ReflectionPad2d((ka, kb, ka, kb)),
            torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)) for i in range(self.num_sim_matrices)])
        
    def forward(self, features):
        results = []
        
        for i in range(self.num_sim_matrices):
            results.append(self.ops[i](features[:,i,:,:,:]).unsqueeze(1))
        
        return torch.cat(results, dim=1)

class CNNLayerV1(torch.nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.kernel_size - 1)//2;
        if self.kernel_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=self.kernel_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = [];
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])));
        return result;
    
class MaxPoolLayerV1(torch.nn.Module):
    def __init__(self, stride, pool_size, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.pool_size = pool_size
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.pool_size - 1)//2;
        if self.pool_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = [];
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])));
        return result;    

class CNNLayer(torch.nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, num_sim_matrices):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sim_matrices = num_sim_matrices
        padding_temp = (self.kernel_size - 1)//2
        if self.kernel_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=self.kernel_size, stride=stride) for i in range(num_sim_matrices)])
        
    def forward(self, features):
        results = []
        
        for i in range(self.num_sim_matrices):
            results.append(self.layers[i](features[:,i,:,:,:]).unsqueeze(1))
        
        return torch.tanh(torch.cat(results, dim=1))

class MaxPoolLayer(torch.nn.Module):
    def __init__(self, stride, pool_size, num_sim_matrices):
#         self.filter = [window_size, window_size, in_channel, out_channel]
        super(MaxPoolLayer, self).__init__()
        self.stride = stride
        self.pool_size = pool_size
        self.num_sim_matrices = num_sim_matrices
        padding_temp = (self.pool_size - 1)//2;
        if self.pool_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=stride) for i in range(num_sim_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = [];
        for i in range(self.num_sim_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])));
        return result;