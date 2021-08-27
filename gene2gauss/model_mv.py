import numpy as np
#import keras
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import backend
from .utils import *
from .layers import *
from .conv_layers import *
from .attn_layers import *
from collections import defaultdict
import math

encoders = {'GCN': GCN,
            'GAT': GAT,
            'multiGCN': multiGCN,
            'multiGAT': multiGAT}

def expand_dims(X, sparse_flag=False):
    if sparse_flag:
        X = tf.sparse.expand_dims(X, axis=0)
    else:
        X = tf.expand_dims(X, axis=0)
    return X

class Gene2Gauss_mv(keras.Model):

    def __init__(self, adjs, adj_2hops, X,
                 L = 64,
                 enc='GCN',
                 reduce = None,
                 n_hidden=[128],
                 symmetric=False,
                 dropout_hid = 0.,
                 dropout_enc = 0.,
                 seed=0,
                 sparse_flag=False,
                 two_hop = False,
                 verbose=True,
                 batch_flag = False,
                 dtype='float32'):
        """
        Multi-view version of Gene2Gauss model
        1. Takes more than one network as input
        2. Learns common gaussian embeddings based on a single (combined) Link prediction objective
        3. Consensus network constructed to identify "consensus" positive edges
        4. Gaussian encoders work on concatenated representations (after non-linear transformations) from each network.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse unweighted adjacency matrix
        X : scipy.sparse.spmatrix
            Sparse attribute matrix
        L : int
            Dimensionality of the node embeddings
        enc: str
            Type of convolution encoder
        agg: str
            method to aggregate node embeddings from each view (concatenate/mean/sum)
        n_hidden : list(int)
            A list specifying the size of each hidden layer, default n_hidden=[512]
        symmetric: bool
            Symmetric/asymmetric KL divergence based loss objective
        verbose : bool
            Verbosity.
        sparse_flag: bool
            Indicating input adjacency matrix type
        two_hop: bool
            Convolutions of two-hop neighborhoods
        batch_flag: bool
            Batch processing
        """
        super(Gene2Gauss_mv, self).__init__()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        X = X.astype(np.float32)
        print('Input Feature shape:',X.shape)

        self.N, self.D = X.shape # N -> number of nodes/genes; D -> input feature dimensions
        self.L = L
        self.symmetric = symmetric

        self.dropout_hid = dropout_hid
        self.dropout_enc = dropout_enc

        self.verbose = verbose
        self.adj_mats = adjs
        self.adj_2hops = adj_2hops
        self.sparse_flag = sparse_flag
        #print("Number of networks:", len(self.adj_mats), "Sparse flag:", self.sparse_flag)
        self.batching = batch_flag
        self.dtype_w = dtype
        self.two_hop = two_hop
        # initializer for weights
        self.w_init = tf.initializers.GlorotUniform
        self.enc_str = enc
        self.enc = encoders[enc]

        self.reduce =reduce
        print("Encoder: ", self.enc, "Reduce:", self.reduce, self.sparse_flag)

        # row normalize the feature matrix
        self.X = preprocess_features(X)

        if n_hidden is None:
            n_hidden = [self.L*2]
        self.n_hidden = n_hidden
        print("***** Number of hidden layers:",len(self.n_hidden), np.max(self.n_hidden))

        if self.sparse_flag:
            self.convert_to_sparse()

        print(self.X.shape, self.adj_mats[0].shape)
        self.__build__()
        #print('Total Number of layers:', len(self.layers_))

    # method to check the number if non-zero elements in the adjacency matrices exceeded the limit
    def check_size(self):
        max_nnz = np.max([adj.nnz for adj in self.adj_mats])
        if self.two_hop:
            max_nnz = np.max([adj.nnz for adj in self.adj_2hops])
        print("Max NNZ:", max_nnz, max_nnz*np.max(self.n_hidden), max_nnz*np.max(self.n_hidden) - math.pow(2,31))
        return max_nnz*np.max(self.n_hidden) - math.pow(2,31) >0

    def expand_dims(self):
        print("*" * 5, "Expanding Dims", "*" * 5)
        if self.sparse_flag:
            self.X = tf.sparse.expand_dims(self.X, axis=0)
        else:
            self.X = tf.expand_dims(self.X, axis=0)

    def convert_to_sparse(self):
        # case where too many non zero entries in the input (sparse) matrix
        # Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]
        if self.check_size() or self.enc_str in ['GAT', 'multiGAT']:
            print("*"*5, "Too many non-zero entries in the inputs, using dense matrix implementation", "*"*5)
            self.X = self.X.todense()
            self.sparse_flag=False
            self.adj_mats = [adj.todense() for adj in self.adj_mats]
            if self.two_hop:
                self.adj_2hops = [adj.todense() for adj in self.adj_2hops]
        else: # converting to sparse tensors
            print("*"*5, "Converting the input networks into sparse tensors", "*"*5)
            self.X = tf.SparseTensor(*sparse_feeder(self.X))
            self.adj_mats = [tf.SparseTensor(*sparse_feeder(adj)) for adj in self.adj_mats]
            self.adj_2hops = [tf.SparseTensor(*sparse_feeder(adj)) for adj in self.adj_2hops]

    # this function initializes the weight matrices (and bias)
    def __build__(self):
        # w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden
        # storing convolution layers for each network
        self.conv_layers = defaultdict(list)
        for network_idx in range(len(self.adj_mats)):
            # adding encoder layers
            for i in range(1, len(sizes)):
                print("*" * 3, "Network Number:", network_idx, "Layer number:", i,
                      "Input Dim:", sizes[i - 1], "Output Dim:", sizes[i], "*" * 3)
                self.conv_layers[network_idx].append(self.enc(sizes[i - 1], sizes[i],
                                                              dropout_rate=self.dropout_hid,
                                                              activation=tf.nn.relu,
                                                              sparse_flag=self.sparse_flag,
                                                              dtype=self.dtype_w))
        self.conv_dim = len(self.adj_mats) * sizes[-1] if self.reduce is None else sizes[-1]
        print('*' * 5, 'Conv Dim:', self.conv_dim, "*" * 5)
        # initiating the gaussian encoder for node embeddings (both mean and variance)
        self.gaussian_encoder = GaussianEncoder(self.conv_dim, self.L,
                                                dropout_rate=self.dropout_enc,
                                                dtype=self.dtype_w)

    #  Training (or validation) over the given set of inputs
    def call(self, inputs):

        # filtering batch specific inputs when used in batch mode
        _, pos_edges, neg_edges, training = inputs
        if training:
            # computing hidden layer compositions
            conv_outputs = defaultdict(list)
            for network_idx in range(len(self.adj_mats)):
                for i in range(len(self.n_hidden)):
                    X = self.X if i==0 else conv_outputs[network_idx][-1]
                    # results from convolutional layers are dense matrices
                    X = tf.SparseTensor(*sparse_feeder(X)) if self.sparse_flag and i>0 else X
                    # expanding dims for GAT implementations
                    X = expand_dims(X, self.sparse_flag) if self.enc_str in ['GAT', 'multiGAT'] else X
                    # cases where two-hop neighborhoods are used
                    if self.enc_str in ['multiGCN', 'multiGAT']:
                        outputs = self.conv_layers[network_idx][i]((X, self.adj_mats[network_idx], self.adj_2hops[network_idx]), training=True)
                    else:
                        outputs = self.conv_layers[network_idx][i]((X,self.adj_mats[network_idx]), training=True)
                    #print("Network: ", network_idx, outputs.shape)
                    conv_outputs[network_idx].append(outputs)

            # final convolved features
            conv_outputs = [conv_outputs[network_idx][-1] for network_idx in range(len(self.adj_mats))]

            # concatenating and reducing the convolved features from all networks
            if self.reduce == None:
                concat_outputs = tf.concat(conv_outputs, axis=1)
            if self.reduce == 'mean':
                #concat_outputs = tf.reduce_mean(tf.concat(conv_outputs, axis=0), axis=1)
                concat_outputs = tf.reduce_mean(conv_outputs, axis=0)
            if self.reduce == 'sum':
                concat_outputs = tf.reduce_sum(conv_outputs, axis=0)
            if self.reduce == 'max':
                concat_outputs = tf.reduce_max(conv_outputs, axis=0)

            self.concat_outputs = concat_outputs
            #computing mu and sigma matrices in the final layers
            self.mu, self.sigma = self.gaussian_encoder(concat_outputs, non_linear = True)
            train_loss = self.compute_train_loss(pos_edges, neg_edges)
            return train_loss

    def link_prediction(self, edges, labels):
        # validation is over a node pairs where the energy represents the unnormalized link prediction score
        link_pred_energy = -self.energy_kl(edges)
        return score_link_prediction(labels, link_pred_energy)

    def compute_train_loss(self, pos_edges, neg_edges):
        # training over positive and negative edges in each batch
        eng_pos = self.energy_kl(pos_edges)
        eng_neg = self.energy_kl(neg_edges)
        if self.symmetric:
            eng_pos += self.energy_kl(tf.stack([pos_edges[:,1], pos_edges[:,0]],1))

        energy = tf.square(eng_pos) + tf.exp(-eng_neg)
        loss = tf.reduce_mean(input_tensor=energy)
        return loss

    def energy_kl(self, pairs):
        """
        Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.
        Based on: https://github.com/abojchevski/graph2gauss

        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated

        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        ij_mu = tf.gather(self.mu, pairs)
        ij_sigma = tf.gather(self.sigma, pairs)

        sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
        trace_fac = tf.reduce_sum(input_tensor=sigma_ratio, axis=1)

        log_det = tf.reduce_sum(input_tensor=tf.math.log(sigma_ratio + 1e-14), axis=1)

        mu_diff_sq = tf.reduce_sum(input_tensor=tf.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], axis=1)

        return 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)
