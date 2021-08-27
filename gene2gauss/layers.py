import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

# global initializer
w_init = tf.initializers.GlorotUniform

# Dropout for sparse Tensors
def sparse_dropout(x, rate, noise_shape):
    # noise_shape = number_nonzero_features
    #print("Noise shape:", noise_shape)
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)
    # Retains specified non-empty values within a SparseTensor
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))

# Encoder layer with a non-linear transformation
class Encoder(layers.Layer):

    def __init__(self, input_dim, output_dim,
                 act=tf.nn.relu,
                 dropout_rate=0.1,
                 dtype = 'float32',
                 **kwargs):
        super(Encoder, self).__init__(dtype=dtype, **kwargs)
        print("    ", "*"*3, "Encoder: input_dim = ", input_dim, "output_dim = ", output_dim, "*"*3)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_rate = dropout_rate

        # initializer for weights
        self.w_init = w_init

        self.W = self.add_weight(name='Weights', shape=(self.input_dim, self.output_dim),
                                      initializer=self.w_init())
        self.b = self.add_weight(name='bias', shape=(self.output_dim,),
                                    initializer=self.w_init())

    def call(self, inputs, non_linear = True, sparse_input = False):
        if sparse_input:
            inputs = sparse_dropout(inputs, self.dropout_rate, self.num_feats_nonzero)
            # multiplying a sparse (features) matrix by a dense (Weights) matrix
            encoded = tf.sparse.sparse_dense_matmul(inputs, self.W) + self.b
        else:
            inputs = tf.nn.dropout(inputs, self.dropout_rate)
            # dense matrix multiplication
            encoded = tf.matmul(inputs, self.W) + self.bs

        if non_linear:
            encoded = self.act(encoded)
        return encoded

# final layer which includes separate weights for mu and sigma
class GaussianEncoder(layers.Layer):
    def __init__(self, input_dim, output_dim,
                 act = tf.nn.elu,
                 dropout_rate=0.1,
                 dtype = 'float32', **kwargs):
        super(GaussianEncoder, self).__init__(dtype=dtype, **kwargs)
        print("    ", "*" * 3, "Gaussian Encoder: input_dim = ", input_dim, "output_dim = ", output_dim, "dropout = ", dropout_rate, "*" * 3)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_rate = dropout_rate

        # initializer for weights
        self.w_init = w_init

        self.weights_mu = self.add_weight(name='Weights_mu', shape=(self.input_dim, self.output_dim), dtype=tf.float32,
                                          initializer=self.w_init())
        self.bias_mu = self.add_weight(name='bias_mu', shape=(self.output_dim,), dtype=tf.float32,
                                       initializer=self.w_init())

        self.weights_sigma = self.add_weight(name='Weights_sigma', shape=(self.input_dim, self.output_dim), dtype=tf.float32,
                                            initializer=self.w_init())
        self.bias_sigma = self.add_weight(name='bias_sigma', shape=(self.output_dim,), dtype=tf.float32,
                                          initializer=self.w_init())

    def call(self, inputs, non_linear = True):
        inputs = tf.nn.dropout(inputs, self.dropout_rate)
        mu = tf.matmul(inputs, self.weights_mu) + self.bias_mu
        sigma = tf.matmul(inputs, self.weights_sigma) + self.bias_sigma
        if non_linear:
            sigma = self.act(sigma) + 1 + 1e-14
        return mu, sigma

# Gated layer
class GATELayer(layers.Layer):
    def __init__(self, in_dim,
                 out_dim,
                 dropout_rate=0.1,
                 activation=tf.nn.relu,
                 initializer='glorot_uniform',
                 regularizer='l2',
                 sparse_flag=True,
                 bias_flag = False):
        super(GATELayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bias_flag = bias_flag
        self.sparse_flag = sparse_flag

        # initializer for weights
        self.w_initializer = initializers.get(initializer)
        # regularizer for weights
        self.w_regularizer = regularizers.get(regularizer)
        self.__build__()

    def __build__(self):
        self.kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                      initializer=self.w_initializer,
                                      regularizer=self.w_regularizer)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        # adding a bias component
        if self.bias_flag:
            self.bias = self.add_weight(name='bias', shape=(self.out_dim,),
                                        initializer=self.w_initializer,
                                        regularizer=self.w_regularizer)

    def call(self, inputs, training=None):
        inputs_1hop = inputs[0]
        inputs_2hop = inputs[1]
        if training:
            inputs_1hop = tf.nn.dropout(inputs_1hop, self.dropout_rate)
            inputs_2hop = tf.nn.dropout(inputs_2hop, self.dropout_rate)

        gate = tf.matmul(inputs_2hop, self.kernel)
        if self.bias_flag:
            gate += self.bias
        gate = tf.nn.tanh(gate)
        if training:
            gate = tf.nn.dropout(gate, self.dropout_rate)
        gate = tf.nn.relu(gate)
        gated_outputs = tf.add(tf.multiply(inputs_1hop, gate), tf.multiply(inputs_2hop, 1-gate))
        return self.activation(gated_outputs)