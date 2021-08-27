import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from .layers import GATELayer

# Dropout for sparse Tensors
def sparse_dropout(x, rate):
    # noise_shape = number_nonzero_features
    noise_shape = x.values.shape
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)
    # Retains specified non-empty values within a SparseTensor
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))

# wrapper for tf.matmul
def dot(x, y, sparse = False):
    if sparse:
        # Multiply SparseTensor (of rank 2) "x" by dense matrix "y".
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# a simple graph convolution layer
class GCN(layers.Layer):
    def __init__(self, in_dim,
                 out_dim,
                 dropout_rate = 0.1,
                 edge_drop = 0.0,
                 activation=tf.nn.relu,
                 initializer = 'glorot_uniform',
                 regularizer = 'l2',
                 sparse_flag = True,
                 bias_flag = False,
                 res_flag = False,
                 dtype='float32',
                 **kwargs):
        '''

        :param in_dim: input (feature) dimension
        :param out_dim: output (embedding) dimension
        :param num_feats_nonzero: num_feats_nonzero (useful for sparse implementation only)
        :param edge_drop: dropout rate for convoluted output (DropEdge). default: 0.0
        :param activation: activation function
        :param initializer: identifier specifying the initializer for kernel weights
        :param regularizer: identifier specifying the regularizer for kernel weights
        :param sparse_flag: boolean flag to indicate the usage of sparse inputs
        :param bias_flag: boolean flag to indicate bias weights
        :param res_flag: boolean flag to indicate if residual connections are neededs
        :param kwargs:
        '''
        super(GCN, self).__init__(dtype=dtype,**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bias_flag = bias_flag
        self.sparse_flag = sparse_flag
        self.res_flag = res_flag

        # initializer for weights
        self.w_initializer = initializers.get(initializer)
        # regularizer for weights
        self.w_regularizer = regularizers.get(regularizer)
        self.__build__()

    def __build__(self):
        # adding the convolution filters
        self.kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                      initializer=self.w_initializer,
                                      regularizer=self.w_regularizer)
        # adding a bias component
        if self.bias_flag:
            self.bias = self.add_weight(name='bias', shape=(self.out_dim,),
                                        initializer=self.w_initializer,
                                        regularizer=self.w_regularizer)

    def call(self, inputs, training=None):
        X, adj = inputs
        # X -> node feature matrix
        # adj -> adjacency matrix
        # dropout
        if training and self.sparse_flag:
            X = sparse_dropout(X, self.dropout_rate)
        elif training:
            X = tf.nn.dropout(X, self.dropout_rate)

        # simple convolution step
        output = dot(X, self.kernel, sparse=self.sparse_flag)
        output = dot(adj, output, sparse=self.sparse_flag)
        # implementing DropEdge mechanism: https://github.com/DropEdge/DropEdge/blob/master/src/models.py
        #output = tf.nn.dropout(output, self.edge_drop)

        # adding the bias component
        if self.bias_flag:
            output += self.bias

        # activation function
        if self.activation is not None:
            output =  self.activation(output)

        return output

class multiGCN(layers.Layer):
    def __init__(self, in_dim,
                 out_dim,
                 dropout_rate=0.1,
                 activation=tf.nn.relu,
                 initializer='glorot_uniform',
                 regularizer='l2',
                 sparse_flag=True,
                 bias_flag=False,
                 dtype='float32',
                 **kwargs):

        '''

        :param in_dim: input (feature) dimension
        :param out_dim: output (embedding) dimension
        :param num_feats_nonzero: num_feats_nonzero (useful for sparse implementation only)
        :param edge_drop: dropout rate for convoluted output (DropEdge). default: 0.0
        :param activation: activation function
        :param initializer: identifier specifying the initializer for kernel weights
        :param regularizer: identifier specifying the regularizer for kernel weights
        :param sparse_flag: boolean flag to indicate the usage of sparse inputs
        :param bias: boolean flag to indicate bias weights
        :param kwargs:
                '''
        super(multiGCN, self).__init__(dtype=dtype, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bias_flag = bias_flag
        self.sparse_flag = sparse_flag
        self.__build__()

    def __build__(self):
        # GCN layer - two-hop neighborhoods
        self.gcn1 = GCN(self.in_dim, self.out_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        sparse_flag=self.sparse_flag)

        # GCN layer - two-hop neighborhoods
        self.gcn2 = GCN(self.in_dim, self.out_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        sparse_flag=self.sparse_flag)

        self.gate = GATELayer(self.out_dim, self.out_dim,
                              dropout_rate=self.dropout_rate,
                              sparse_flag=self.sparse_flag,
                              bias_flag=self.bias_flag)

    def call(self, inputs, training=None):
        X, adj_1hop, adj_2hop = inputs
        # X -> node feature matrix
        # adj -> adjacency matrix
        outputs_1hop = self.gcn1((X, adj_1hop), training=training)
        outputs_2hop = self.gcn2((X, adj_2hop), training=training)
        outputs_gate = self.gate((outputs_1hop, outputs_2hop), training=training)
        return outputs_gate