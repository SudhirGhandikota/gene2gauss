import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf

class MiniBatchIterator(object):
    '''
    This iterator class generates consensus positive and negative edges associated with each node
    For each iteration it generates a batch of nodes along with positive and negative edges for training
    '''
    def __init__(self, adjs,
                 train_edges=None,
                 pre_flag=False,
                 sparse_flag = False,
                 batch_size=512,
                 pos_cnt = 1,
                 neg_cnt = 1,
                 **kwargs):
        '''
        :param adjs: list of adjacency matrices
        :param train_edges: pre-compiled training edges
        :param pre_flag: boolean flag indicating if pre-compiledd edges are being used
        :param sparse_flag: type of input adjacency matrix
        :param batch_size: number of nodes in a batch
        :param p_val: percentage of validation edges
        :param p_test: percentage of testing edges
        :param kwargs:
        '''

        self.adjs = adjs
        self.pre_flag = pre_flag
        if train_edges is not None:
            self.pos_edges = train_edges[0]
            self.neg_edges = train_edges[1]

        # converting sparse matrices to dense versions for constructing combined adjacency
        if sparse_flag:
            self.adjs = [np.array(adj.todense()) for adj in self.adjs]

        # computing a consensus adjacency network
        self.combined_adj = np.max(self.adjs, axis=0)
        self.combined_adj = normalize(self.combined_adj, axis=1, norm='l1')

        self.num_nodes = self.adjs[0].shape[0]
        self.nodes = np.array(list(range(self.adjs[0].shape[0])))

        self.batch_size = batch_size
        self.pos_cnt = pos_cnt
        self.neg_cnt = neg_cnt
        self.batch_counter = 0

    # this function extracts the nodes for the next batch
    def next_batch(self):
        # extracting the nodes for the next batch
        start_idx = self.batch_counter * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_nodes)
        batch_nodes = self.nodes[start_idx:end_idx]
        self.batch_counter += 1
        if self.pre_flag:
            pos_edges, neg_edges = self.filter_train_edges(batch_nodes)
        else:
            pos_edges, neg_edges = self.filter_edges(batch_nodes)

        # resetting batch counter
        if end_idx >= self.num_nodes:
            self.batch_counter = 0
            self.nodes = np.random.permutation(self.nodes)
        return tf.convert_to_tensor(batch_nodes), tf.convert_to_tensor(pos_edges), tf.convert_to_tensor(neg_edges)

    # this function randomly generates positive and negative training edges for the next batch
    def filter_edges(self, batch_nodes):
        pos_edges = []
        neg_edges = []
        for node_id in batch_nodes:
            # positive edges sampled based on probabilities (normalized correlations)
            pos_edges.append((node_id,
                              np.random.choice(self.combined_adj.shape[0], self.pos_cnt, p= self.combined_adj[node_id])[0]))

            # negative edges sampled randomly
            neg_edges.append((node_id,
                              np.random.choice(np.where(self.combined_adj[node_id]==0)[0], self.neg_cnt)[0]))
        return pos_edges, neg_edges

    # this function randomly generates positive and negative training edges for all nodes
    def filter_edges_all(self):
        pos_edges = []
        neg_edges = []
        nodes = np.random.permutation(self.nodes)
        for node_id in nodes:
            # positive edges sampled based on probabilities (normalized correlations)
            pos_edges.append((node_id, np.random.choice(self.combined_adj.shape[0], self.pos_cnt, p=self.combined_adj[node_id])[0]))

            # negative edges sampled randomly
            neg_edges.append((node_id, np.random.choice(np.where(self.combined_adj[node_id] == 0)[0], 1)[0]))
        return pos_edges, neg_edges

# this function randomly generates positive and negative training edges for all nodes
    def filter_train_edges(self, batch_nodes):
        # filtering positive edges containing the batch nodes
        pos_edges = np.unique(
            np.concatenate((self.pos_edges[np.in1d(self.pos_edges[:, 0], batch_nodes)],
                            self.pos_edges[np.in1d(self.pos_edges[:, 1], batch_nodes)])),
            axis=0)
        neg_edges = np.unique(
            np.concatenate((self.neg_edges[np.in1d(self.neg_edges[:, 0], batch_nodes)],
                            self.neg_edges[np.in1d(self.neg_edges[:, 1], batch_nodes)])),
            axis=0)
        min_len = min(len(pos_edges), len(neg_edges))
        return pos_edges[:min_len], neg_edges[:min_len]