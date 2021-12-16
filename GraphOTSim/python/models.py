import random
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv
from lap import lapjv
from lap import lapmod

from torch_geometric.nn import GCNConv
from GraphOTSim.external.PairNorm.layers import PairNorm
from GraphOTSim.python.layers import Conv2dSameLayer, AttentionModule, MaxPool2dLayer, CNNLayer, CNNLayerV1, MaxPoolLayer, MaxPoolLayerV1, TensorNetworkModule

def greedy_ot(cost_matrix, return_matching=False):
    class cost_node:
        def __init__(self, i, j, cost):
            self.i = i;
            self.j = j;
            self.cost = cost;
            self.up = None;
            self.down = None;
            self.left = None;
            self.right = None;
            self.prev = None;
            self.next = None;
        def __lt__(self, other):
            return self.cost < other.cost
        def delete(self):
            if self.right:
                self.right.left = self.left;
            if self.left:
                self.left.right = self.right;
            if self.up:
                self.up.down = self.down;
            if self.down:
                self.down.up = self.up;
        def __str__(self):
            return str((self.i,self.j, self.cost))
    num_pts = len(cost_matrix);
    C_cpu = cost_matrix.detach().cpu().numpy();
    # create cost nodes for every possible matching
    cost_node_list = np.array([np.array([cost_node(i,j,C_cpu[i,j]) for j in range(num_pts)]) for i in range(num_pts)]);
    # Add location details
    for i in range(num_pts):
        for j in range(num_pts):
            if i > 0:
                cost_node_list[i,j].up = cost_node_list[i-1,j];
            if j > 0:
                cost_node_list[i,j].left = cost_node_list[i, j-1];
            if i+1 < num_pts:
                cost_node_list[i,j].down = cost_node_list[i+1, j];
            if j+1 < num_pts:
                cost_node_list[i,j].right = cost_node_list[i, j+1];
    # make 1D and sort
    sorted_cost_list = np.sort(cost_node_list, axis=None)
    # make linked list of cost_nodes
    for i in range(0, num_pts**2-1):
        sorted_cost_list[i].next = sorted_cost_list[i+1];
    for i in range(1, num_pts**2):
        sorted_cost_list[i].prev = sorted_cost_list[i-1];
    head_node = cost_node(None,None,None);
    head_node.next = sorted_cost_list[0];
    sorted_cost_list[0].prev = head_node;
    tail_node = cost_node(None,None,None);
    tail_node.prev = sorted_cost_list[-1];
    sorted_cost_list[-1].next = tail_node;
    col_ind = [-1]*num_pts;
    row_ind = [-1]*num_pts;
    # Start the magic
    while head_node.next != tail_node:
        min_cost_node = head_node.next;
        row = min_cost_node.i;
        col = min_cost_node.j;
        col_ind[row] = col;
        row_ind[col] = row;
        #Delete same row nodes
        #Left
        current = min_cost_node;
        while True:
            if current.left == None:
                break;
            current = current.left;
            current.prev.next, current.next.prev = current.next, current.prev;
            current.delete()
        #Right
        current = min_cost_node;
        while True:
            if current.right == None:
                break;
            current = current.right;
            current.prev.next, current.next.prev = current.next, current.prev;
            current.delete()
        #Up
        current = min_cost_node;
        while True:
            if current.up == None:
                break;
            current = current.up;
            current.prev.next, current.next.prev = current.next, current.prev;
            current.delete()
        #down
        current = min_cost_node;
        while True:
            if current.down == None:
                break;
            current = current.down;
            current.prev.next, current.next.prev = current.next, current.prev;
            current.delete()
        head_node.next = min_cost_node.next;
        head_node.next.prev = head_node;
        min_cost_node.delete()
    loss = torch.tensor([0.0]).cuda();
    for i in range(num_pts):
        loss += cost_matrix[i,col_ind[i]];
    if return_matching:
        return loss/num_pts, (col_ind, row_ind);
    else:
        return loss/num_pts;

def compute_frobenius_pairwise_distances_torch(X, Y, device, p=1, normalized=True):
    """Compute pairwise distances between 2 sets of points"""
    assert X.shape[1] == Y.shape[1]
    
    d = X.shape[1]
    dists = torch.zeros(X.shape[0], Y.shape[0], device=device)

    for i in range(X.shape[0]):
        if p == 1:
            dists[i, :] = torch.sum(torch.abs(X[i, :] - Y), dim=1)
        elif p == 2:
            dists[i, :] = torch.sum((X[i, :] - Y) ** 2, dim=1)
        else:
            raise Exception('Distance type not supported: p={}'.format(p))
        
        if normalized:
            dists[i, :] = dists[i, :] / d

    return dists

EPSILON = 1e-12
def dense_wasserstein_distance(cost_matrix, scaling=False, return_matching=False):
    num_pts = len(cost_matrix);
    C_cpu = cost_matrix.detach().cpu().numpy();    
    if scaling:
        C_cpu *= 100000 / (C_cpu.max() + EPSILON)
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(C_cpu);
    
    loss = torch.tensor([0.0]).cuda();
    for i in range(num_pts):
        loss += cost_matrix[i,col_ind_lapjv[i]];
                
    if return_matching:
        return loss/num_pts, (col_ind_lapjv, row_ind_lapjv);
    else:
        return loss/num_pts;

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
#     m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def is_pos_def(M):
    return np.all(np.linalg.eigvals(M) > 0)

from torch.autograd import Variable
from torch.autograd import Function

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        eps=1e-10
        
        m = input.detach().cpu().numpy().astype(np.float_)
        if not is_pos_def(m):
            msg = ('not SPD; '
               'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(m.shape[0]) * eps
            m = m + offset
            if not is_pos_def(m):
                raise Exception('not SPD')
        sqrtm_cpu = scipy.linalg.sqrtm(m).real
        if not np.isfinite(sqrtm_cpu).all():
            msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sqrtm_cpu.shape[0]) * eps
            sqrtm_cpu = linalg.sqrtm(m + offset)
        
        sqrtm = torch.from_numpy(sqrtm_cpu).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_variables
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float_)
            gm = grad_output.detach().cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)

sqrtm = MatrixSquareRoot.apply

def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = Variable(torch.eye(dim,dim).view(1, dim, dim).
                 repeat(batchSize,1,1).type(dtype),requires_grad=False).cuda()
    Z = Variable(torch.eye(dim,dim).view(1, dim, dim).
                 repeat(batchSize,1,1).type(dtype),requires_grad=False).cuda()

    for i in range(numIters):
       U = 0.5*(3.0*I - Z.bmm(Y))
       Y = Y.bmm(U)
       Z = U.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    # error = compute_error(A, sA)
    return sA, 0

def compute_g_FID_loss_by_iterative(real_features, gen_features, normalized=True):
    d = real_features.size(1)

    mu1 = torch.mean(real_features, dim=0)
    mu2 = torch.mean(gen_features, dim=0)
    mu_diff = mu1 - mu2
    
    sigma1 = cov(real_features)
    sigma2 = cov(gen_features)
    A = torch.matmul(sigma1, sigma2)
    sqrtA, error = sqrt_newton_schulz_autograd(A.unsqueeze(dim=0), 10, A.dtype)


    mean_loss = mu_diff.dot(mu_diff)
    cov_loss = torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(sqrtA.squeeze())
    gloss = mean_loss + cov_loss
    if normalized:
        gloss = gloss / d
    return gloss

class GeometricGraphSim(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
    
    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
        
        self.gcn_layers = torch.nn.ModuleList([]);
        self.conv_layers = torch.nn.ModuleList([]);
        self.pool_layers = torch.nn.ModuleList([]);
        self.linear_layers = torch.nn.ModuleList([]);        
        self.num_conv_layers = len(self.args.conv_kernel_size);
        self.num_linear_layers = len(self.args.linear_size);
                
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
        
        in_channels = 1;
        for i in range(self.num_conv_layers):
            self.conv_layers.append(CNNLayerV1(kernel_size=self.args.conv_kernel_size[i], stride=1, in_channels=in_channels, out_channels=self.args.conv_out_channels[i], num_similarity_matrices=self.num_gcn_layers))
            self.pool_layers.append(MaxPoolLayerV1(pool_size=self.args.conv_pool_size[i], stride=self.args.conv_pool_size[i], num_similarity_matrices=self.num_gcn_layers))
            in_channels = self.args.conv_out_channels[i];
            
        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(self.args.linear_size[i], self.args.linear_size[i+1]));
            
        self.scoring_layer = torch.nn.Linear(self.args.linear_size[-1], 1)

    def graph_convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param graph: DGL graph.
        :param features: Feature matrix.
        :return features: List of abstract feature matrices.
        """
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            if self.pairnorm:
                features = self.pairnorm(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
            

        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def image_convolutional_pass(self, similarity_matrices_list):
        """
        Making convolutional pass with 2D conv layers.
        :param similarity_matrices_list: List of similarity matrices between the nodes of the two graphs.
        :return features: List of abstract feature matrices.
        """
        features = [_.unsqueeze(0).unsqueeze(0) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features);
    
            features = [torch.nn.functional.dropout(_,
                                               p=self.args.dropout,
                                               training=self.training)  for _ in features]
        
        return features
    
    def linear_pass(self, features):
        """
        Making pass with the final feed forward network.
        :param features, resized output of 2D conv.
        :return features: GED estimate
        """
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = torch.nn.functional.relu(features);
            features = torch.nn.functional.dropout(features,p=self.args.dropout,
                                               training=self.training)
        return features
    
    def forward(self, data):
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()
        
        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
        
        bfs_order_1 = torch.cat(dgl.bfs_nodes_generator(G_1, random.randint(0, n1-1)))
        bfs_order_2 = torch.cat(dgl.bfs_nodes_generator(G_2, random.randint(0, n2-1)))
        
        abstract_features_list_1 = [_[bfs_order_1] for _ in abstract_features_list_1];
        abstract_features_list_2 = [_[bfs_order_2] for _ in abstract_features_list_2];
        
        similarity_matrices_list = [torch.mm(abstract_features_list_1[i], abstract_features_list_2[i].transpose(0,1)) for i in range(self.num_gcn_layers)]
        if n1 > n2:
            padding_length = n1 - n2;
            p1d = (0, padding_length)
            similarity_matrices_list = [F.pad(_, p1d, "constant", 0) for _ in similarity_matrices_list];
            
        elif n2 > n1:
            padding_length = n2 - n1;
            p1d = (0, 0, 0, padding_length)
            similarity_matrices_list = [F.pad(_, p1d, "constant", 0) for _ in similarity_matrices_list];
            
        similarity_matrices_list = [F.interpolate(_.unsqueeze(0).unsqueeze(0), size=self.args.interpolate_size, align_corners=True, mode=self.args.interpolate_mode).squeeze(0).squeeze(0) for _ in similarity_matrices_list];
        
        features = torch.cat(self.image_convolutional_pass(similarity_matrices_list)).view(-1, self.args.linear_size[0])
        features = self.linear_pass(features);
        
        
        score_logits = self.scoring_layer(features)
        score = torch.sigmoid(score_logits)
        return score.view(-1), score_logits.view(-1)
  
cosine_similarity = torch.nn.CosineSimilarity(dim=1)

def compute_pairwise_distances_torch(X, Y, dist_type='cosine', normalized=False):
    """Compute pairwise distances between 2 sets of points"""
    if len(Y.shape) == 1:
        Y = Y.view([1, -1]) #convert to mat of [1, dim]
        
    d = X.shape[1]
    assert d == Y.shape[1]    

    dists = torch.zeros(X.shape[0], Y.shape[0], dtype=X.dtype)
    
    if X.is_cuda:
        dists = dists.cuda()

#     for i in range(X.shape[0]):
#         if dist_type == 'l1':
#             dists[i, :] = torch.sum(torch.abs(X[i, :] - Y), dim=1)
#         elif dist_type == 'l2':
#             dists[i, :] = torch.sum((X[i, :] - Y) ** 2, dim=1)
#         elif dist_type == 'cosine':
#             dists[i, :] = -cosine_similarity(X[[i], :], Y)
#         else:
#             raise Exception('Distance type not supported: p={}'.format(p))
#         if normalized:
#             dists[i, :] = dists[i, :] / d
            
    for i in range(Y.shape[0]):
        if dist_type == 'l1':
            dists[:, i] = torch.sum(torch.abs(X - Y[i, :]), dim=1)
        elif dist_type == 'l2':
            dists[:, i] = torch.sum((X - Y[i, :]) ** 2, dim=1)
        elif dist_type == 'cosine':
            dists[:, i] = -cosine_similarity(X, Y[[i], :])
        else:
            raise Exception('Distance type not supported: p={}'.format(p))

        if normalized:
            dists[:, i] = dists[:, i] / d

    #make sure to output a vector if Y is vector
    return dists.view(X.shape[0], Y.shape[0]) if Y.shape[0] > 1 else dists 

class GOTSim(GeometricGraphSim):
    """
    OT cost on graphsim
    """
    def __init__(self, args, number_of_labels):
        super().__init__(args, number_of_labels)    
    
    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
            
        self.gcn_layers = torch.nn.ModuleList([])
        
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
        
        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)
        
        # Params for insertion and deletion embeddings 
        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        for i in range(self.num_gcn_layers):
            self.insertion_params.append(torch.nn.Parameter(torch.ones(self.args.gcn_size[i])))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.args.gcn_size[i])))

    def forward(self, data, return_matching=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()


        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
                
        if 'distance_type' in self.args:
            if self.args.distance_type == 'negative_dot':
                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]                    
            elif self.args.distance_type == 'batch_cosine': #this is for whole batch version of the cosine distance
                abstract_features_list_1 = [F.normalize(abstract_features_list_1[i], dim=1) for i in range(self.num_gcn_layers)]
                abstract_features_list_2 = [F.normalize(abstract_features_list_2[i], dim=1) for i in range(self.num_gcn_layers)]
                
                deletion_params = [F.normalize(self.deletion_params[i], dim=0) for i in range(self.num_gcn_layers)]
                insertion_params = [F.normalize(self.insertion_params[i], dim=0) for i in range(self.num_gcn_layers)]

                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)
                ]
            else:
                main_similarity_matrices_list = [compute_pairwise_distances_torch(
                                                      abstract_features_list_1[i], 
                                                      abstract_features_list_2[i],
                                                      self.args.distance_type) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_1[i], self.deletion_params[i], self.args.distance_type)) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_2[i], self.insertion_params[i], self.args.distance_type)) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]
        else: #this is original, keep it here for sanity check later
            main_similarity_matrices_list = [torch.mm(abstract_features_list_1[i], 
                                                      abstract_features_list_2[i].transpose(0,1)) 
                                             for i in range(self.num_gcn_layers)]


            # these are matrix with 0 on the diagonal and inf cost on off-diagonal
            insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n1))).cuda()
            deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n2))).cuda()

            deletion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                 + insertion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]

            insertion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                 + deletion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]            
        
        dummy_similarity_matrices_list = [
            torch.zeros(n2, n1, dtype=abstract_features_list_1[i].dtype).cuda()
            for i in range(self.num_gcn_layers)
        ]
            
        similarity_matrices_list = [
            torch.cat(
                (
                    torch.cat((main_similarity_m, deletion_similarity_m), dim=1),
                    torch.cat((insertion_similarity_m, dummy_similarity_m), dim=1)
                ), dim=0)
            for main_similarity_m, deletion_similarity_m, insertion_similarity_m, dummy_similarity_m 
            in zip(main_similarity_matrices_list, deletion_similarity_matrices_list, 
                   insertion_similarity_matrices_list, dummy_similarity_matrices_list)
        ]
        
        matching = [dense_wasserstein_distance(s, scaling=False, return_matching=return_matching) 
                         for s in similarity_matrices_list]
        if return_matching:
            matching_cost = [e[0] for e in matching] #first element
            matches = [e[1] for e in matching]
        else:
            matching_cost = matching
        
        if 'matching_type' in self.args:
            if self.args.matching_type == 'last': #only on the last matching
                score_logits = matching_cost[-1]
                score = torch.sigmoid(score_logits)
        else:
            matching_cost = 2 * torch.cat(matching_cost) / (n1 + n2)        
            score_logits = self.ot_scoring_layer(matching_cost)
            score = torch.sigmoid(score_logits)
            
        if return_matching:
            return score, score_logits, matches;
        else:
            return score, score_logits    
        
class UnnormalizedGOTSim(GOTSim):
    """
    OT cost on graphsim
    """
    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
            
        self.gcn_layers = torch.nn.ModuleList([])
        
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
        
        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)
        
        # Params for insertion and deletion embeddings 
        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        for i in range(self.num_gcn_layers):
            self.insertion_params.append(torch.nn.Parameter(torch.ones(self.args.gcn_size[i])))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.args.gcn_size[i])))
    
    def forward(self, data, return_matching=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()


        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
                
        if 'distance_type' in self.args:
            if self.args.distance_type == 'negative_dot':
                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]                    
            elif self.args.distance_type == 'batch_cosine': #this is for whole batch version of the cosine distance
                abstract_features_list_1 = [F.normalize(abstract_features_list_1[i], dim=1) for i in range(self.num_gcn_layers)]
                abstract_features_list_2 = [F.normalize(abstract_features_list_2[i], dim=1) for i in range(self.num_gcn_layers)]
                
                deletion_params = [F.normalize(self.deletion_params[i], dim=0) for i in range(self.num_gcn_layers)]
                insertion_params = [F.normalize(self.insertion_params[i], dim=0) for i in range(self.num_gcn_layers)]

                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)
                ]
            else:
                main_similarity_matrices_list = [compute_pairwise_distances_torch(
                                                      abstract_features_list_1[i], 
                                                      abstract_features_list_2[i],
                                                      self.args.distance_type) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_1[i], self.deletion_params[i], self.args.distance_type)) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_2[i], self.insertion_params[i], self.args.distance_type)) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]
        else: #this is original, keep it here for sanity check later
            main_similarity_matrices_list = [torch.mm(abstract_features_list_1[i], 
                                                      abstract_features_list_2[i].transpose(0,1)) 
                                             for i in range(self.num_gcn_layers)]


            # these are matrix with 0 on the diagonal and inf cost on off-diagonal
            insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n1))).cuda()
            deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n2))).cuda()

            deletion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                 + insertion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]

            insertion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                 + deletion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]            
        
        dummy_similarity_matrices_list = [
            torch.zeros(n2, n1, dtype=abstract_features_list_1[i].dtype).cuda()
            for i in range(self.num_gcn_layers)
        ]
            
        similarity_matrices_list = [
            torch.cat(
                (
                    torch.cat((main_similarity_m, deletion_similarity_m), dim=1),
                    torch.cat((insertion_similarity_m, dummy_similarity_m), dim=1)
                ), dim=0)
            for main_similarity_m, deletion_similarity_m, insertion_similarity_m, dummy_similarity_m 
            in zip(main_similarity_matrices_list, deletion_similarity_matrices_list, 
                   insertion_similarity_matrices_list, dummy_similarity_matrices_list)
        ]
        
        matching = [dense_wasserstein_distance(s, scaling=False, return_matching=return_matching) 
                         for s in similarity_matrices_list]
        if return_matching:
            matching_cost = [e[0] for e in matching] #first element
            matches = [e[1] for e in matching]
        else:
            matching_cost = matching
        
        if 'matching_type' in self.args:
            if self.args.matching_type == 'last': #only on the last matching
                score_logits = matching_cost[-1]
                score = torch.sigmoid(score_logits)
        else:
#             matching_cost = torch.cat(matching_cost)
#             matching_cost = torch.relu(self.linear_layer(matching_cost))
            matching_cost = torch.cat(matching_cost)
            score_logits = self.ot_scoring_layer(matching_cost)
            score = torch.sigmoid(score_logits)
            
        if return_matching:
            return score, score_logits, matches;
        else:
            return score, score_logits            

class UnnormalizedGOTSimWithLinear(GOTSim):
    """
    OT cost on graphsim
    """
    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
            
        self.gcn_layers = torch.nn.ModuleList([])
        
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
        
        self.calculate_bottleneck_features()
        self.linear_layer = torch.nn.Linear(self.feature_count, 2)
        
        
        self.ot_scoring_layer = torch.nn.Linear(2, 1)
        
        # Params for insertion and deletion embeddings 
        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        for i in range(self.num_gcn_layers):
            self.insertion_params.append(torch.nn.Parameter(torch.ones(self.args.gcn_size[i])))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.args.gcn_size[i])))

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            if 'histogram_all' in self.args and self.args.histogram_all:
                self.feature_count = self.num_gcn_layers + self.args.bins * self.num_gcn_layers
            else:
                self.feature_count = self.num_gcn_layers + self.args.bins
        else:
            self.feature_count = self.num_gcn_layers

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        #print(abstract_features_1.shape, abstract_features_2.shape)
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist
    
    def forward(self, data, return_matching=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()


        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
                
        if 'distance_type' in self.args:
            if self.args.distance_type == 'negative_dot':
                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]                    
            elif self.args.distance_type == 'batch_cosine': #this is for whole batch version of the cosine distance
                abstract_features_list_1 = [F.normalize(abstract_features_list_1[i], dim=1) for i in range(self.num_gcn_layers)]
                abstract_features_list_2 = [F.normalize(abstract_features_list_2[i], dim=1) for i in range(self.num_gcn_layers)]
                
                deletion_params = [F.normalize(self.deletion_params[i], dim=0) for i in range(self.num_gcn_layers)]
                insertion_params = [F.normalize(self.insertion_params[i], dim=0) for i in range(self.num_gcn_layers)]

                main_similarity_matrices_list = [-torch.mm(abstract_features_list_1[i], 
                                                          abstract_features_list_2[i].transpose(0,1)) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_1[i], deletion_params[i])) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(-torch.matmul(abstract_features_list_2[i], insertion_params[i])) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)
                ]
            else:
                main_similarity_matrices_list = [compute_pairwise_distances_torch(
                                                      abstract_features_list_1[i], 
                                                      abstract_features_list_2[i],
                                                      self.args.distance_type) 
                                                 for i in range(self.num_gcn_layers)]


                # these are matrix with 0 on the diagonal and inf cost on off-diagonal
                insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n1))).cuda()
                deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                                - torch.diag(torch.ones(n2))).cuda()

                deletion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_1[i], self.deletion_params[i], self.args.distance_type)) 
                     + insertion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]

                insertion_similarity_matrices_list = [
                    torch.diag(compute_pairwise_distances_torch(abstract_features_list_2[i], self.insertion_params[i], self.args.distance_type)) 
                     + deletion_constant_matrix
                    for i in range(self.num_gcn_layers)

                ]
        else: #this is original, keep it here for sanity check later
            main_similarity_matrices_list = [torch.mm(abstract_features_list_1[i], 
                                                      abstract_features_list_2[i].transpose(0,1)) 
                                             for i in range(self.num_gcn_layers)]


            # these are matrix with 0 on the diagonal and inf cost on off-diagonal
            insertion_constant_matrix = 99999 * (torch.ones(n1, n1, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n1))).cuda()
            deletion_constant_matrix = 99999 * (torch.ones(n2, n2, dtype=abstract_features_list_1[0].dtype) 
                                            - torch.diag(torch.ones(n2))).cuda()

            deletion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_1[i], self.deletion_params[i])) 
                 + insertion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]

            insertion_similarity_matrices_list = [
                torch.diag(torch.matmul(abstract_features_list_2[i], self.insertion_params[i])) 
                 + deletion_constant_matrix
                for i in range(self.num_gcn_layers)

            ]            
        
        dummy_similarity_matrices_list = [
            torch.zeros(n2, n1, dtype=abstract_features_list_1[i].dtype).cuda()
            for i in range(self.num_gcn_layers)
        ]
            
        similarity_matrices_list = [
            torch.cat(
                (
                    torch.cat((main_similarity_m, deletion_similarity_m), dim=1),
                    torch.cat((insertion_similarity_m, dummy_similarity_m), dim=1)
                ), dim=0)
            for main_similarity_m, deletion_similarity_m, insertion_similarity_m, dummy_similarity_m 
            in zip(main_similarity_matrices_list, deletion_similarity_matrices_list, 
                   insertion_similarity_matrices_list, dummy_similarity_matrices_list)
        ]
        
        matching = [dense_wasserstein_distance(s, scaling=False, return_matching=return_matching) 
                         for s in similarity_matrices_list]
        if return_matching:
            matching_cost = [e[0] for e in matching] #first element
            matches = [e[1] for e in matching]
        else:
            matching_cost = matching
        
        if 'matching_type' in self.args:
            if self.args.matching_type == 'last': #only on the last matching
                score_logits = matching_cost[-1]
                score = torch.sigmoid(score_logits)
        else:
            matching_cost = torch.cat(matching_cost).view(1, -1)
            
            
            if self.args.histogram == True:
                if 'histogram_all' in self.args and self.args.histogram_all:
                    hist = [self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2)) for (abstract_features_1, abstract_features_2) in zip(abstract_features_list_1, abstract_features_list_2)]
                    matching_cost = torch.cat([matching_cost] + hist, dim=1).view(1, -1)
                else:
                    abstract_features_1 = abstract_features_list_1[-1]
                    abstract_features_2 = abstract_features_list_2[-1]
                    hist = self.calculate_histogram(abstract_features_1,
                                                    torch.t(abstract_features_2))
                    matching_cost = torch.cat((matching_cost, hist), dim=1).view(1, -1)
            
            matching_cost = torch.relu(self.linear_layer(matching_cost))
            score_logits = self.ot_scoring_layer(matching_cost)
            score = torch.sigmoid(score_logits)
            
        if return_matching:
            return score.view(-1), score_logits.view(-1), matches;
        else:
            return score.view(-1), score_logits.view(-1)
        
class FrechetSim(GOTSim):
    """
    OT cost on graphsim
    """
    def __init__(self, args, number_of_labels):
        super().__init__(args, number_of_labels)
    
    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None
            
        self.gcn_layers = torch.nn.ModuleList([])
        
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];

        self.linear_layer1 = torch.nn.Linear(self.num_gcn_layers, self.num_gcn_layers)
        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)
        
    def forward(self, data, return_matching=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()


        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
                
        matching_cost = [2 * compute_g_FID_loss_by_iterative(f1, f2, normalized=True) / (n1 + n2) for f1, f2 in zip(abstract_features_list_1, abstract_features_list_2)]
        matching_cost = torch.tensor(matching_cost).cuda()
        
        if 'matching_type' in self.args:
            if self.args.matching_type == 'last': #only on the last matching
                score_logits = matching_cost[-1]
                score = torch.sigmoid(score_logits)
        else:
            matching_cost = torch.cat(matching_cost)
            matching_cost = self.linear_layer1(matching_cost)
            score_logits = self.ot_scoring_layer(matching_cost)
            score = torch.sigmoid(score_logits)
            
        if return_matching:
            return score, score_logits, matches;
        else:
            return score, score_logits          
class SimGNNReadoutPaper(GeometricGraphSim):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation." WSDM. 2019.
    http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf
    """
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            if 'pairnorm_version' in self.args:
                pairnorm_version = self.args.pairnorm_version
            else:
                pairnorm_version = 'PN'
            if 'pairnorm_scale' in self.args:
                pairnorm_scale = self.args.pairnorm_scale
            else:
                pairnorm_scale = 1
            print(f'Use pairnorm: {pairnorm_version}, scale {pairnorm_scale}')
            self.pairnorm = PairNorm(mode=pairnorm_version, scale=pairnorm_scale)
        else:
            self.pairnorm = None
                
        self.calculate_bottleneck_features()
        self.gated_layer = torch.nn.Linear(self.args.gcn_size[-1],
                                                     self.args.gcn_size[-1])
        self.gcn_layers = torch.nn.ModuleList([]);
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
            
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected = torch.nn.Linear(self.feature_count,
                                                    self.args.bottle_neck_neurons)
        self.num_linear_layers = len(self.args.linear_size);
        self.linear_layers = torch.nn.ModuleList([])
        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(self.args.linear_size[i], self.args.linear_size[i+1]));
        self.scoring_layer = torch.nn.Linear(self.args.linear_size[-1], 1)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()
        
        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)

        abstract_features_1 = abstract_features_list_1[-1]
        abstract_features_2 = abstract_features_list_2[-1]
        if self.args.readout == "max":
            pooled_features_1 = torch.max(abstract_features_1, dim=0, keepdim=True)[0].transpose(0,1)
            pooled_features_2 = torch.max(abstract_features_2, dim=0, keepdim=True)[0].transpose(0,1)
        elif self.args.readout == "mean":
            pooled_features_1 = torch.mean(abstract_features_1, dim=0, keepdim=True).transpose(0,1)
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0,1)
        elif self.args.readout == "gated":
            pooled_features_1 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_1))*abstract_features_1, dim=0, keepdim=True).transpose(0,1)
            pooled_features_2 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_2))*abstract_features_2, dim=0, keepdim=True).transpose(0,1)
        features = self.tensor_network(pooled_features_1, pooled_features_2).view(1, -1)
        features = self.linear_pass(features)        
        score_logit = self.scoring_layer(features)
        score = torch.sigmoid(score_logit)
        return score.view(-1), score_logit.view(-1)
    
class SimGNN(GeometricGraphSim):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation." WSDM. 2019.
    http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf
    """        
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        if 'use_pairnorm' in self.args and self.args.use_pairnorm: 
            print('Use pairnorm')
            self.pairnorm = PairNorm()
        else:
            self.pairnorm = None

        self.calculate_bottleneck_features()

        self.gcn_layers = torch.nn.ModuleList([]);
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i];
            
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()

        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)
        
        abstract_features_1 = abstract_features_list_1[-1]
        abstract_features_2 = abstract_features_list_2[-1]

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score_logits = self.scoring_layer(scores)
        score = torch.sigmoid(score_logits)
        return score.view(-1), score_logits.view(-1)     
    
def to_numpy(l):
    if type(l) == list:
        return [_.detach().cpu().numpy() for _ in l]
    else:
        return l.detach().cpu().numpy()
     
class GMN(SimGNN):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation." WSDM. 2019.
    http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf
    """        
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.gated_layer = torch.nn.Linear(self.args.gcn_size[-1],
                                                     self.args.gcn_size[-1])
#         num_ftrs = self.number_labels;
#         self.num_gcn_layers = len(self.args.gcn_size);
#         self.gcn_layers = torch.nn.ModuleList([]);
#         self.gcn_update_wights = torch.nn.ModuleList([]);
#         for i in range(self.num_gcn_layers):
#             self.gcn_layers.append(GraphConv(in_feats=num_ftrs, out_feats=self.args.gcn_size[i], activation=None, weight=True, bias=False))
#             self.gcn_update_wights.append(torch.nn.Linear(num_ftrs*2+self.args.gcn_size[i], self.args.gcn_size[i]));
#             num_ftrs = self.args.gcn_size[i];
            
        num_ftrs = self.number_labels;
        self.num_gcn_layers = len(self.args.gcn_size);
        self.gcn_layers = torch.nn.ModuleList([]);
        self.gcn_update_wights = torch.nn.ModuleList([]);
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                GCNConv(num_ftrs, self.args.gcn_size[i], bias=False))
            self.gcn_update_wights.append(torch.nn.Linear(num_ftrs*2+self.args.gcn_size[i], self.args.gcn_size[i]));
            num_ftrs = self.args.gcn_size[i];

            
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
    def graph_convolutional_pass(self, edge_index_1, edge_index_2, features_1, features_2):
        """
        Making convolutional pass.
        :param graph: DGL graph.
        :param features: Feature matrix.
        :return features: List of abstract feature matrices.
        """
        for i in range(self.num_gcn_layers-1):
            # print(features_1.shape, features_2.shape)
            conv_1_output = self.gcn_layers[i](features_1, edge_index_1) #self.gcn_layers[i](graph_1, features_1);
            conv_2_output = self.gcn_layers[i](features_2, edge_index_2) #self.gcn_layers[i](graph_2, features_2);
            
            if self.args.similarity == "cosine":
                similarity_matrix = torch.mm(F.normalize(features_1, dim=1), F.normalize(features_2, dim=1).transpose(0,1));
            elif self.args.similarity == "euclidean":
                similarity_matrix = -torch.cdist(features_1, features_2);
            elif self.args.similarity == "dot":
                similarity_matrix = torch.mm(features_1, features_2.transpose(0,1));
            a_1 = torch.softmax(similarity_matrix, dim=1)
            a_2 = torch.softmax(similarity_matrix, dim=0)
            attention_1 = torch.mm(a_1, features_2)
            attention_2 = torch.mm(a_2.transpose(0,1), features_1)
            features_1 = self.gcn_update_wights[i](torch.cat([conv_1_output, features_1, features_1 - attention_1], dim=1));
            features_2 = self.gcn_update_wights[i](torch.cat([conv_2_output, features_2, features_2 - attention_2], dim=1));
            features_1 = torch.tanh(features_1)
            features_2 = torch.tanh(features_2)
            features_1 = torch.nn.functional.dropout(features_1,
                                               p=self.args.dropout,
                                               training=self.training)
            features_2 = torch.nn.functional.dropout(features_2,
                                               p=self.args.dropout,
                                               training=self.training)
        # print(features_1.shape, features_2.shape)
        conv_1_output = self.gcn_layers[-1](features_1, edge_index_1) #self.gcn_layers[-1](graph_1, features_1);
        conv_2_output = self.gcn_layers[-1](features_2, edge_index_2) #self.gcn_layers[-1](graph_2, features_2);
        
        
        if self.args.similarity == "cosine":
            similarity_matrix = torch.mm(F.normalize(features_1, dim=1), F.normalize(features_2, dim=1).transpose(0,1));
        elif self.args.similarity == "euclidean":
            similarity_matrix = -torch.cdist(features_1, features_2);
        elif self.args.similarity == "dot":
            similarity_matrix = torch.mm(features_1, features_2.transpose(0,1));
        a_1 = torch.softmax(similarity_matrix, dim=1)
        a_2 = torch.softmax(similarity_matrix, dim=0)
        attention_1 = torch.mm(a_1, features_2)
        attention_2 = torch.mm(a_2.transpose(0,1), features_1)
        features_1 = self.gcn_update_wights[-1](torch.cat([conv_1_output, features_1, features_1 - attention_1], dim=1));
        features_2 = self.gcn_update_wights[-1](torch.cat([conv_2_output, features_2, features_2 - attention_2], dim=1));
        return features_1, features_2;


    def forward(self, data):
        edge_index_1 = data["edge_index_1"].cuda()
        edge_index_2 = data["edge_index_2"].cuda()
        
        G_1, G_2 = data['G_1'], data['G_2']        
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()

        abstract_features_1, abstract_features_2 = self.graph_convolutional_pass(edge_index_1, edge_index_2, features_1, features_2)
        
        if self.args.readout == "max":
            pooled_features_1 = torch.max(abstract_features_1, dim=0, keepdim=True)[0].transpose(0,1)
            pooled_features_2 = torch.max(abstract_features_2, dim=0, keepdim=True)[0].transpose(0,1)
        elif self.args.readout == "mean":
            pooled_features_1 = torch.mean(abstract_features_1, dim=0, keepdim=True).transpose(0,1)
            pooled_features_2 = torch.mean(abstract_features_2, dim=0, keepdim=True).transpose(0,1)
        elif self.args.readout == "gated":
            pooled_features_1 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_1))*abstract_features_1, dim=0, keepdim=True).transpose(0,1)
            pooled_features_2 = torch.sum(torch.sigmoid(self.gated_layer(abstract_features_2))*abstract_features_2, dim=0, keepdim=True).transpose(0,1)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.tanh(self.fully_connected_first(scores))
        score_logit = self.scoring_layer(scores)
        score = torch.sigmoid(score_logit)
        return score.view(-1), score_logit.view(-1)    
    
                    
