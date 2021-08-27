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

# Single GAT layer (single attention head)
class GAT(layers.Layer):
    def __init__(self, in_dim,
                 out_dim,
                 dropout_rate=0.1,
                 activation=tf.nn.elu,
                 sparse_flag=True,
                 bias_flag=False,
                 res_flag=False,
                 dtype='float32',
                 **kwargs):
        '''
        :param in_dim: input (feature) dimension - unused in this layer but added the parameter for usability reasons
        :param out_dim: output (embedding) dimension
        :param dropout: dropout rate. default: 0.0
        :param activation: activation function
        :param sparse_flag: boolean flag to indicate the usage of sparse inputs
        :param bias: boolean flag to indicate bias weights
        :param res_flag: boolean flag to indicate if residual connections are needed
        :param kwargs:
        '''
        super(GAT, self).__init__(dtype=dtype, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.res_flag = res_flag
        self.bias_flag = bias_flag
        self.sparse_flag = sparse_flag
        self.dropout_rate = dropout_rate

        ### convolution operations
        # 1. shared linear transformation -> num_filters = hidden dimension; filter_size = 1
        self.conv_shared = layers.Conv1D(self.out_dim, 1, use_bias=self.bias_flag)
        # 2. self-attention
        self.conv1 = layers.Conv1D(1, 1)
        self.conv2 = layers.Conv1D(1, 1)
        # optional residual step
        self.conv_res = layers.Conv1D(self.out_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(self.out_dim))

    def sparse_call(self, inputs, training=None):
        X, adj = inputs
        if training:
            X = sparse_dropout(X, self.dropout_rate)
        seq_fts = self.conv_shared(X)

        # simple attention operation -> equivalent of a single layer feedforward neural network parameterized by a weight vector of size 2F'
        f1 = self.conv1(seq_fts)
        f2 = self.conv2(seq_fts)

        f1 = tf.reshape(f1, (X.shape[0], 1))
        f2 = tf.reshape(f2, (X.shape[0], 1))
        f1 = adj*f1
        f2 = adj*tf.transpose(f2, [1,0])

        logits = tf.sparse.add(f1, f2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        # attention coefficients
        coefs = tf.sparse.softmax(lrelu)
        # sparse dropout
        if training:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs, self.dropout_rate),
                                    dense_shape=coefs.dense_shape)
            seq_fts = tf.nn.dropout(seq_fts, self.dropout_rate)

        coefs = tf.sparse.reshape(coefs, [X.shape[0], X.shape[0]])
        seq_fts = tf.squeeze(seq_fts)

        # new features = linear combination of features using normalized attention coefficients
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, X.shape[0], self.out_dim])

        rets = vals + self.bias_zero

        # adding residual connections
        if self.res_flag:
            if X.shape[-1] != rets.shape[-1]:
                rets = rets + self.conv_res(X)
            else:
                rets = rets + X
        rets = self.activation(rets)

        # removing the additional dimension for concatenation
        rets = tf.squeeze(rets)
        # implementing DropEdge mechanism: https://github.com/DropEdge/DropEdge/blob/master/src/models.py
        #rets = tf.nn.dropout(rets, self.edge_drop)

        return self.activation(rets)

   # attention step from GAT: https://github.com/PetarV-/GAT
    def call(self, inputs, training=None):
        if self.sparse_flag:
            self.sparse_call(inputs, training)
        X, adj = inputs
        # X -> node feature matrix
        # adj -> adjacency matrix
        if training:
            X = tf.nn.dropout(X, self.dropout_rate)
        seq_fts = self.conv_shared(X)

        # simple attention operation -> equivalent of a single layer feedforward neural network parameterized by a weight vector of size 2F'
        f1 = self.conv1(seq_fts)
        f2 = self.conv2(seq_fts)
        logits = f1 + tf.transpose(f2, [0, 2, 1])

        # attention coefficients
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits)+adj)

        if training:
            coefs = tf.nn.dropout(coefs, self.dropout_rate)
            seq_fts = tf.nn.dropout(seq_fts, self.dropout_rate)

        # new features -> linear combination of features using the normalized attention coefficients
        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        rets = vals + self.bias_zero

        # adding residual connections
        if self.res_flag:
            if X.shape[-1] != rets.shape[-1]:
                rets = rets + self.conv_res(X)
            else:
                rets = rets + X
        rets = self.activation(rets)
        # removing the additional dimension for concatenation
        rets = tf.squeeze(rets)
        return self.activation(rets)

# multiheaded GAT layer
class multiheadGAT(layers.Layer):
    def __init__(self,
                 hid_dim,
                 num_heads=4,
                 dropout_rate=0.1,
                 activation = tf.nn.elu,
                 sparse_flag=True,
                 bias_flag=False,
                 res_flag=False, **kwargs):
        '''
        :param out_dim: output (embedding) dimension
        :param dropout: dropout rate. default: 0.0
        :param activation: activation function
        :param sparse_flag: boolean flag to indicate the usage of sparse inputs
        :param bias: boolean flag to indicate bias weights
        :param kwargs:
        '''
        super(multiheadGAT, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.sparse_flag = sparse_flag
        self.bias_flag = bias_flag
        self.res_flag = res_flag

        self.attns = []

        # individual attention heads
        for i in range(num_heads):
            self.attns.append(GAT(self.hid_dim,
                                  dropout_rate=self.dropout_rate,
                                  activation=self.activation,
                                  sparse_flag=self.sparse_flag,
                                  bias_flag=self.bias_flag,
                                  res_flag=self.res_flag))

    def call(self, inputs, training=None):
        X, adj = inputs
        outs = []

        # individual attention heads
        for attn in self.attns:
            outs.append(attn((X, adj)))

        h_outs = tf.concat(outs, axis=-1)
        return h_outs

# GAT layer with multihop attention
class multiGAT(layers.Layer):
    def __init__(self, in_dim,
                 out_dim,
                 dropout_rate=0.1,
                 activation = tf.nn.elu,
                 sparse_flag=True,
                 bias_flag=False,
                 res_flag=True,
                 dtype='float32',
                 **kwargs):
        '''
        :param in_dim: input (feature) dimension - unused in this layer but added the parameter for usability reasons
        :param out_dim: output (embedding) dimension
        :param dropout: dropout rate. default: 0.0
        :param activation: activation function
        :param sparse_flag: boolean flag to indicate the usage of sparse inputs
        :param bias_flag: boolean flag to indicate bias weights
        :param res_flag: boolean flag to indicate if residual connections are needed
        :param kwargs:
        '''
        super(multiGAT, self).__init__(dtype=dtype, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.sparse_flag = sparse_flag
        self.bias_flag = bias_flag
        self.res_flag = res_flag
        self.__build__()

    def __build__(self):
        # adding GAT layers for each hop
        # GCN layer - two-hop neighborhoods
        self.gat1 = GAT(self.in_dim, self.out_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        sparse_flag=self.sparse_flag,
                        res_flag=self.res_flag)

        # GCN layer - two-hop neighborhoods
        self.gat2 = GAT(self.in_dim, self.out_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        sparse_flag=self.sparse_flag,
                        res_flag=self.res_flag)

        self.gate = GATELayer(self.out_dim, self.out_dim,
                              dropout_rate=self.dropout_rate,
                              sparse_flag=self.sparse_flag,
                              bias_flag=self.bias_flag)

    def call(self, inputs, training=None):
        X, adj = inputs
        outs = []

        # individual attention heads
        for attn in self.attns:
            outs.append(attn((X, adj)))

        h_outs = tf.concat(outs, axis=-1)
        return h_outs

    def call(self, inputs, training=None):
        X, adj_1hop, adj_2hop = inputs
        # X -> node feature matrix
        # adj -> adjacency matrix
        outputs_1hop = self.gat1((X, adj_1hop), training=training)
        outputs_2hop = self.gat2((X, adj_2hop), training=training)
        outputs_gate = self.gate((outputs_1hop, outputs_2hop), training=training)
        return outputs_gate