import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
import itertools
import os
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
import nvidia_smi

def get_memory_info():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total_memory = round(((memory_info.total/1024)/1024)/1024,3)
    free_memory = round(((memory_info.free / 1024) / 1024) / 1024,3)
    used_memory = round(((memory_info.used / 1024) / 1024) / 1024,3)
    print("*"*5, "Total Memory (GB):", total_memory, "Used Memory (GB):", used_memory, "Free Memory (GB):", free_memory, "*"*5)

def preprocess_gcn(adj_mats, sparse=False):
    print("*" * 5, "Preprocessing of input networks", "*" * 5)
    if sparse:
        adj_mats = [sp.csr_matrix(adj) for adj in adj_mats]

    adj_mats = [preprocess_adj(adj, sparse) for adj in adj_mats]
    return adj_mats

def preprocess_gat(adj_mats, num_nodes, sparse=False):
    print("*" * 5, "Preprocessing of input networks", "*" * 5)
    adj_mats = [adj_to_bias(adj, num_nodes, sparse) for adj in adj_mats]
    return adj_mats

def normalize_adj(adj, add_loops=False):
    """Symmetrically normalize adjacency matrix."""
    if add_loops:
        np.fill_diagonal(adj, 1.0)
    #print("Min:", np.min(adj))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj, sparse_flag=True, loops=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if sparse_flag:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = adj_normalized.tocoo()
    else:
        adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized
    #return sparse_to_tuple(adj_normalized)

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, num_nodes, sparse, nhood=1):
    mt = np.eye(adj.shape[1]) # (2708, 2708) identity matrix
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[1])))
    mt[mt>=0.5] = 1.0
    mt[mt<0.5] = 0.0
    mt = -1e9 * (1.0 - mt)
    if sparse:
        mt = sp.csr_matrix(mt)
    else:
        mt = mt[np.newaxis]
    return mt

# row normalize feature matrix
def preprocess_features(features):
    rowsum = np.array(features.sum(1)) # get sum of each row
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum,
    r_inv[np.isinf(r_inv)] = 0. # zeroing inf indices
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix,
    features = r_mat_inv.dot(features) # D^-1
    return features

def to_tuple(mx):
    # check if the matrix is a coo(coordinated list) style sparse matrix
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose() # coordinates i.e. positions where sparse matrix has non-zero values
    values = mx.data # actual data
    shape = mx.shape
    return coords, values, shape

# this function converts sparse matrix to tuple representation
def sparse_to_tuple(sparse_mx):
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# this function loads the adjacencies matrices from a given input directory
def load_adjacencies(inpath, file_type='npz', sparse_flag=False, keep_weighted=True):
    adj_mats = []
    files = os.listdir(inpath)
    for file in files:
        print("*"*10, "Loading", file, "*"*10)
        if sparse_flag:
            adj = sp.load_npz(os.path.join(inpath, file))
            # removing self-edges
            adj.setdiag(0)
        else:
            adj = np.load(os.path.join(inpath, file), allow_pickle=True)
            if file_type == 'npz':  # in case if its a archived numpy file
                adj = adj['arr_0']
            # removing self edges
            np.fill_diagonal(adj, 0.0)

        # converting to an unweighted network
        if not keep_weighted:
            adj[adj>0]=1
         # removing negative weighted edges if any
        adj[adj < 0] = 0.0
        adj_mats.append(adj)

    print("*"*3, "Number of networks:", len(adj_mats), "Number of nodes:",adj_mats[0].shape[0], "*"*3)
    return adj_mats

def load_test_edges(infile, random_state=42):
    data = np.load(infile)
    if '.npz' in infile:
        train_edges = data['arr_0']
        train_labels = data['arr_1']
        test_edges = data['arr_2']
        test_labels = data['arr_3']
    '''
    skf = StratifiedKFold(n_splits=2)
    for val_index, test_index in skf.split(edges, labels):
        val_edges, val_labels = edges[val_index], labels[val_index]
        test_edges, test_labels = edges[test_index], labels[test_index] '''
    train_edges, val_edges, train_labels, val_labels = train_test_split(train_edges, train_labels, random_state=random_state)
    return train_edges, val_edges, test_edges, train_labels, val_labels, test_labels

# this function returns train, val and test edges for a list of input adjacency networks
def split_adjacencies(adj_mats, p_val=0.2, p_test=0.2, seed=42, num_neg=1, threshold=0.8):
    # train_pos_edges, val_pos_edges, test_pos_edges, train_neg_edges, val_neg_edges, test_neg_edges = split_adjacency(adj_mats[0])
    # fetching train, valid and test edges for each network
    edges_info = [(split_adjacency(adj_mats[idx], p_val=p_val, p_test=p_test, seed=seed, num_neg=num_neg, weighted_threshold=threshold))
                  for idx in range(len(adj_mats))]

    # unique train, valid and test edges combined from all networks
    all_train_pos_edges = set().union(*[edges_info[idx][0] for idx in range(len(adj_mats))])
    all_val_pos_edges = set().union(*[edges_info[idx][1] for idx in range(len(adj_mats))])
    all_test_pos_edges = set().union(*[edges_info[idx][2] for idx in range(len(adj_mats))])

    all_train_neg_edges = set().union(*[edges_info[idx][3] for idx in range(len(adj_mats))])
    all_val_neg_edges = set().union(*[edges_info[idx][4] for idx in range(len(adj_mats))])
    all_test_neg_edges = set().union(*[edges_info[idx][5] for idx in range(len(adj_mats))])
    '''
    print("All Positive edges(train):", len(all_train_pos_edges),
          "All positive edges (validation):", len(all_val_pos_edges),
          "All positive edges (test):", len(all_test_pos_edges),
          "All negative edges(train):", len(all_train_neg_edges),
          "All negative edges (validation):", len(all_val_neg_edges),
          "All negative edges (test):", len(all_test_neg_edges)) '''

    all_train_edges = np.array(list(all_train_pos_edges) + list(all_train_neg_edges))
    all_train_labels = np.array([1]*len(all_train_pos_edges) + [0]*len(all_train_neg_edges))
    all_val_edges = np.array(list(all_val_pos_edges) + list(all_val_neg_edges))
    all_val_labels = np.array([1] * len(all_val_pos_edges) + [0] * len(all_val_neg_edges))
    all_test_edges = np.array(list(all_test_pos_edges) + list(all_test_neg_edges))
    all_test_labels = np.array([1] * len(all_test_pos_edges) + [0] * len(all_test_neg_edges))

    print("All training edges:", all_train_edges.shape, np.bincount(all_train_labels),
          "All validation edges:", all_val_edges.shape, np.bincount(all_val_labels),
          "All test edges:", all_test_edges.shape, np.bincount(all_test_labels))
    return all_train_edges, all_train_labels, all_val_edges, all_val_labels, all_test_edges, all_test_labels

# this function splits a given adjacency matrix into train, valid and test edges
def split_adjacency(adj, p_val=0.2, p_test=0.2, seed=42, num_neg=1, weighted_threshold = 0.8):
    """
    Parameters:
    :param adj: input adjacency matrix
    :param p_val: percentage of validation edges needed
    :param p_test: percedtage of test edges needed
    :param seed: seed for random computations
    :param num_neg: Number of negative samples to be generated for each positive edge
    :param weighted_threshold: threshold value for weighted adjacency values to be considered for positive edges
    :return: training, validation and test edges (both positive and negative)
    """

    # generating positive edges
    rows, cols = np.where(adj >= weighted_threshold)
    pos_indices = [(rows[idx], cols[idx]) for idx in range(rows.shape[0])]
    # adjusting test edge proportions
    p_test =p_test*(1/(1-p_val))
    train_pos_edges, val_pos_edges = train_test_split(pos_indices, test_size=p_val, random_state=seed)
    train_pos_edges, test_pos_edges = train_test_split(train_pos_edges, test_size=p_test, random_state=seed)

    # generating negative edges
    rows, cols = np.where(adj == 0)
    random_indices = np.random.choice(rows.shape[0], len(pos_indices)*num_neg)
    neg_indices = [(rows[idx], cols[idx]) for idx in random_indices]
    train_neg_edges, val_neg_edges = train_test_split(neg_indices, test_size=p_val, random_state=seed)
    train_neg_edges, test_neg_edges = train_test_split(train_neg_edges, test_size=p_test, random_state=seed)

    return train_pos_edges, val_pos_edges, test_pos_edges, train_neg_edges, val_neg_edges, test_neg_edges


def sparse_feeder(M):
    """
    Prepares the input matrix into a format that is easy to feed into tensorflow's SparseTensor

    Parameters
    ----------
    M : scipy.sparse.spmatrix
        Matrix to be fed

    Returns
    -------
    indices : array-like, shape [n_edges, 2]
        Indices of the sparse elements
    values : array-like, shape [n_edges]
        Values of the sparse elements
    shape : array-like
        Shape of the matrix
    """
    M = sp.coo_matrix(M)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def cartesian_product(x, y):
    """
    Form the cartesian product (i.e. all pairs of values) between two arrays.
    Parameters
    ----------
    x : array-like, shape [Nx]
        Left array in the cartesian product
    y : array-like, shape [Ny]
        Right array in the cartesian product

    Returns
    -------
    xy : array-like, shape [Nx * Ny]
        Cartesian product

    """
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def score_link_prediction(labels, scores):
    """
    Calculates the area under the ROC curve and the average precision score.

    Parameters
    ----------
    labels : array-like, shape [N]
        The ground truth labels
    scores : array-like, shape [N]
        The (unnormalized) scores of how likely are the instances

    Returns
    -------
    roc_auc : float
        Area under the ROC curve score
    ap : float
        Average precision score
    """
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def score_node_classification(features, z, p_labeled=0.1, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.

    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm

    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    lrcv = LogisticRegressionCV()

    if norm:
        features = normalize(features)

    trace = []
    for seed in range(n_repeat):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        lrcv.fit(features[split_train], z[split_train])
        predicted = lrcv.predict(features[split_test])

        f1_micro = f1_score(z[split_test], predicted, average='micro')
        f1_macro = f1_score(z[split_test], predicted, average='macro')

        trace.append((f1_micro, f1_macro))

    return np.array(trace).mean(0)


def get_k_hops(A, K, keep_weighted = True):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : The graph represented as an adjacency matrix
    K : The maximum number of hops to consider.
    keep_weighted: To keep the k-hop network as a weighted network
    sparse_flag: To use sparse matrices as inputs

    Returns
    -------
    hops : the kth hop neighborhood (key -1 is used to accumulate all neighborhoods)
    """
    #print("One-hop:", A.nnz)
    #A = A.todense()
    all_hops = [A]
    for h in range(1,K):
        print('*'*3, "Computing hop:", h+1, "*"*3)
        next_hop = all_hops[h - 1].dot(A)
        #print(next_hop)

        if not keep_weighted:
            next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(0, h):
            next_hop -= np.multiply(next_hop, all_hops[prev_h])

        next_hop[next_hop < 0] = 0
        #next_hop.setdiag(0)
        np.fill_diagonal(next_hop, 0.0)
        next_hop = sp.coo_matrix(next_hop)
        all_hops.append(next_hop)
        #print("Two-hop:", next_hop.nnz)
    return all_hops[-1]

def get_hops_sparse(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hops to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """

    hops = {1: A.tolil()}
    hops[1].setdiag(0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = hops[h - 1].dot(A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= next_hop.multiply(hops[prev_h])

        next_hop = next_hop.tolil()
        next_hop.setdiag(0)

        hops[h] = next_hop

    return hops

def get_hops_dense(A, K):
    """
    Calculates the K-hop neighborhoods of the nodes in a graph.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The graph represented as a sparse matrix
    K : int
        The maximum hops to consider.

    Returns
    -------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    """
    A = A.todense()
    hops = {1: A}
    np.fill_diagonal(hops[1], 0.0)

    for h in range(2, K + 1):
        # compute the next ring
        next_hop = np.dot(hops[h - 1], A)
        next_hop[next_hop > 0] = 1

        # make sure that we exclude visited n/edges
        for prev_h in range(1, h):
            next_hop -= np.multiply(next_hop, hops[prev_h])

        np.fill_diagonal(next_hop, 0.0)
        hops[h] = sp.csr_matrix(next_hop).tolil()

    return hops

def sample_last_hop(A, nodes):
    """
    For each node in nodes samples a single node from their last (K-th) neighborhood.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix encoding which nodes belong to any of the 1, 2, ..., K-1, neighborhoods of every node
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N]
        The sampled nodes.
    """
    N = A.shape[0]

    sampled = np.random.randint(0, N, len(nodes))

    nnz = A[nodes, sampled].nonzero()[1]
    while len(nnz) != 0:
        new_sample = np.random.randint(0, N, len(nnz))
        sampled[nnz] = new_sample
        nnz = A[nnz, new_sample].nonzero()[1]

    return sampled


def sample_all_hops(hops, nodes=None):
    """
    For each node in nodes samples a single node from all of their neighborhoods.

    Parameters
    ----------
    hops : dict
        A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
    nodes : array-like, shape [N]
        The nodes to consider

    Returns
    -------
    sampled_nodes : array-like, shape [N, K]
        The sampled nodes.
    """

    N = hops[1].shape[0]
    if nodes is None:
        nodes = np.arange(N)
    # stacking together randomly picked neighbors at each level for every node
    return np.vstack((nodes,
                      np.array([[-1 if len(x) == 0 else np.random.choice(x) for x in hops[h].rows[nodes]]
                                for h in hops.keys() if h != -1]),
                      sample_last_hop(hops[-1], nodes)
                      )).T

def calc_kl(p_mu, q_mu, p_sigma, q_sigma):
    """
    Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.

    Parameters
    ----------
    pairs : array-like, shape [?, 2]
         The edges/non-edges for which the energy is calculated

     Returns
    -------
    energy : array-like, shape [?]
        The energy of each pair given the currently learned model
    """
    # print(p_mu.shape, q_mu.shape, p_sigma.shape, q_sigma.shape)
    L = p_mu.shape[0]
    sigma_ratio = p_sigma / q_sigma
    # print("sigma_ratio:", sigma_ratio.shape)
    trace_fac = sigma_ratio.sum(axis=0)
    # print("trace_fac:", trace_fac)

    log_det = np.log(sigma_ratio + 1e-14).sum(axis = 0)
    # print("log_det:", log_det)

    mu_diff_sq = np.square(p_mu - q_mu) / p_sigma
    mu_diff_sq = mu_diff_sq.sum(axis = 0)
    # print("mu_diff_sq:", mu_diff_sq)

    return 0.5 * (trace_fac + mu_diff_sq - L - log_det)