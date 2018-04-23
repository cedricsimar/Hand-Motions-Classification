# -*- coding: utf-8 -*-

from tf_decorator import *
from settings import Settings

import tensorflow as tf
from tensorflow.python.ops import array_ops as tf_array_ops


class RCNN:

    def __init__(self, input_ph, is_training):

        # receiving input placeholder
        self.input = input_ph

        # boolean placeholder for batch normalization (BN) "training" parameter
        # BN has beta 
        self.is_training = is_training  

        # Network settings
        self.norm_eps = Settings.NORMALIZATION_EPSILON

        # getting the number of hand motions to classify
        self.output_size = Settings.NUM_EVENTS

        # weights and biases dictionary
        self.learning_parameters = {}
        self.layers = {}

        # initialize tensorflow graph
        self.predict
        self.optimize
        self.error

        # Initialize input placeholder to assign values to weights and biases
        with tf.variable_scope("input_assignment"):

            self.l_param_input = {}
            self.assign_operator = {}
            for variable_name in self.learning_parameters.keys():
                self.l_param_input[variable_name] = tf.placeholder(
                    tf.float32,
                    self.learning_parameters[variable_name].get_shape().as_list(),
                    name=variable_name)

                try:  # If mutable tensor (Variable)
                    self.assign_operator[variable_name] = self.learning_parameters[variable_name].assign(
                        self.l_param_input[variable_name])
                except AttributeError as e:
                    print(e)

    
    def recurrent_convolutionnal_layer(self, conv_t0):
        """
        Unfolded approximation (T = 3) of a recurrent network using convolutional layers
        t = 0: input (feed-forward convolution)
        t = 1, 2, 3 unfolded approximation
        
        Tensorflow note: use "reuse" in t2 and t3 to share the weights of t1
        """
        
        conv_t1 = tf.layers.conv2d(inputs=conv_t0, filters=256, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None, reuse=None)
        
        sum_t1 = tf.add(conv_t1, conv_t0)
        bn_t1 = tf.layers.batch_normalization(inputs=sum_t1, epsilon=self.norm_eps)
        nl_t1 = tf.nn.leaky_relu(bn_t1)


        conv_t2 = tf.layers.conv2d(inputs=nl_t1, filters=256, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None, reuse=True)
        
        sum_t2 = tf.add(conv_t2, conv_t0)
        bn_t2 = tf.layers.batch_normalization(inputs=sum_t2, epsilon=self.norm_eps)
        nl_t2 = tf.nn.leaky_relu(bn_t2)


        conv_t3 = tf.layers.conv2d(inputs=nl_t2, filters=256, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None, reuse=True)
        
        sum_t3 = tf.add(conv_t3, conv_t0)
        bn_t3 = tf.layers.batch_normalization(inputs=sum_t3, epsilon=self.norm_eps)
        nl_t3 = tf.nn.leaky_relu(bn_t3)

        return(nl_t3)


    @define_scope
    def predict(self):
        """
        The input to the neural network consists of a 32 channels x SAMPLE_LENGTH signal 
        produced by the preprocessing stage
        """
        # reshape input to 3d tensor [batch, channels, sample length]
        input_layer = tf.reshape(self.input,
                                [-1, Settings.NUM_CHANNELS, Settings.SAMPLE_LENGTH, 1])


        """
        Layer 1: 1D temporal convolution over the raw signal of each channel 

        2 approaches: 
            1) convolution -> batch normalization -> non-linearity -> max-pooling -> dropout (tested)
            2) convolution -> non-linearity -> batch normalization -> max-polling -> dropout (in the paper)
        """

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=256, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None)
        
        bn1 = tf.layers.batch_normalization(inputs=conv1, epsilon=self.norm_eps)
        nl1 = tf.nn.leaky_relu(bn1)
        pool1 = tf.layers.max_pooling2d(inputs=nl1, pool_size=(1, 4), strides=(1, 4))
        # no dropout in the first layer

        """
        Layer 2: Recurrent convolutionnal layer 
        """
        conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[1, 1], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None)
        
        bn2 = tf.layers.batch_normalization(inputs=conv2, epsilon=self.norm_eps)
        nl2 = tf.nn.leaky_relu(bn2)
        rcl2 = self.recurrent_convolutionnal_layer(nl2) # pool1, conv2 or nl2 ???
        pool2 = tf.layers.max_pooling2d(inputs=rcl2, pool_size=(1, 4), strides=(1, 4))
        drop2 = tf.layers.dropout(inputs=pool2, rate=Settings.DROPOUT_RATE)


        """
        Layer 3: Recurrent convolutionnal layer 
        """
        conv3 = tf.layers.conv2d(inputs=drop2, filters=256, kernel_size=[1, 1], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None)
        
        bn3 = tf.layers.batch_normalization(inputs=conv3, epsilon=self.norm_eps)
        nl3 = tf.nn.leaky_relu(bn3)
        rcl3 = self.recurrent_convolutionnal_layer(nl3) # pool2, conv3 or nl3 ???
        pool3 = tf.layers.max_pooling2d(inputs=rcl3, pool_size=(1, 4), strides=(1, 4))
        drop3 = tf.layers.dropout(inputs=pool3, rate=Settings.DROPOUT_RATE)


        """
        Layer 4: Recurrent convolutionnal layer 
        """
        conv4 = tf.layers.conv2d(inputs=drop3, filters=256, kernel_size=[1, 1], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None)
        
        bn4 = tf.layers.batch_normalization(inputs=conv4, epsilon=self.norm_eps)
        nl4 = tf.nn.leaky_relu(bn4)
        rcl4 = self.recurrent_convolutionnal_layer(nl4) # pool3, conv4 or nl4 ???
        pool4 = tf.layers.max_pooling2d(inputs=rcl4, pool_size=(1, 4), strides=(1, 4))
        drop4 = tf.layers.dropout(inputs=pool4, rate=Settings.DROPOUT_RATE)
        

        """
        Layer 5: Recurrent convolutionnal layer 
        """
        conv5 = tf.layers.conv2d(inputs=drop4, filters=256, kernel_size=[1, 1], strides=(1, 1),
                                 padding = "same", kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                                 activation = None)
        
        bn5 = tf.layers.batch_normalization(inputs=conv5, epsilon=self.norm_eps)
        nl5 = tf.nn.leaky_relu(bn5)
        rcl5 = self.recurrent_convolutionnal_layer(nl5) # pool3, conv4 or nl4 ???
        pool5 = tf.layers.max_pooling2d(inputs=rcl5, pool_size=(1, 2), strides=(1, 2))
        # no dropout before the output layer


        """
        Layer 6: Output layer
        """
        logits = tf.layers.dense(inputs=pool5, units=Settings.NUM_EVENTS,
                               kernel_initializer = tf.init_ops.glorot_uniform_initializer,
                               activation = None)

        # using sigmoid cross entropy (not mutually exclusive) with logits so no need of an
        # activation function at the end of the RCNN
        
        return(logits)


    @define_scope
    def optimize(self):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate = Settings.LEARNING_RATE,
                momentum = Settings.MOMENTUM,
                use_nesterov = True
            )
            return(self.optimizer.minimize(self.loss))


    @define_scope
    def errors(self):
        """ Return the clipped errors """
        # placeholders for the target network q values and the action
        self.target_q = tf.placeholder(tf.float32, [None], name="target_q")
        self.action = tf.placeholder(tf.int64, [None], name="action")

        # convert the action to one-hot representation in order to compute the
        # error
        action_one_hot = tf.one_hot(
            self.action,
            self.action_space,
            on_value=1,
            off_value=0,
            name="action_one_hot")

        self.q_acted = tf.reduce_sum(
            self.q_values *
            tf.cast(
                action_one_hot,
                tf.float32),
            axis=1,
            name="q_acted")

        self.delta = self.target_q - self.q_acted

        """
        [Article] We also found it helpful to clip the error term from the update r + gamma max_d Q(s', a', theta-)
        to be between -1 and 1. Because the absolute value loss function |x| has a derivative of -1
        for all negative values of x and a derivative of 1 for all positive values of x,
        clipping the squared error to be between -1 and 1 corresponds to using an absolute value
        loss function for errors outside of the (-1,1) interval. This form of error clipping further
        improved the stability of the algorithm.

        It is called the Huber loss and because the name is so cool, we have to implement it
        With d = 1 (we could also try with d = 2) (d <> self.delta)
        x =  0.5 * x^2                  if |x| <= d
        x =  0.5 * d^2 + d * (|x| - d)  if |x| > d
        """
        self.clipped_error = tf_array_ops.where(tf.abs(self.delta) < 1.0,
                                                tf.square(self.delta) * 0.5,
                                                tf.abs(self.delta) - 0.5)
        return(self.clipped_error)

    @define_scope
    def error(self):
        """ Return the mean (clipped) error """
        self.mean_error = tf.reduce_mean(self.errors, name="mean_error")
        return(self.mean_error)

    @define_scope
    def importance_weighted_error(self):
        """ Return the importance-weighted error for prioritized memory updates """
        weighted_errors = self.i_s_weights * self.errors
        self.mean_error = tf.reduce_mean(weighted_errors, name="mean_error")
        return(self.mean_error)

    def weight_variable(self, shape, method="normal"):
        """
        Initialize weight variable randomly using one of the two following methods:
            - A truncated normal distribution of mean = 0 and standard deviation of 0.02
            - A xavier initialization, where var(W) = 1 / n_in
        """
        assert(method in ["normal", "xavier"])
        if method == "normal":
            weight_var = tf.truncated_normal(shape, mean=0, stddev=0.02)
        else:
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            weight_var = xavier_initializer(list(shape))
        return(tf.Variable(weight_var))

    def bias_variable(self, shape):
        """
        Initialize bias variables of a specific shape using a constant
        """
        bias_var = tf.constant(0.1, shape=shape)
        return(tf.Variable(bias_var))

    def get_value(self, var_name, tf_session):
        """
        Return the value of the tf variable named [var_name] if it exists, None otherwise
        """

        if var_name in self.learning_parameters:

            value = tf_session.run(self.learning_parameters[var_name])

        elif var_name in self.layers:

            value = tf_session.run(self.layers[var_name])

        else:
            print("Unknown DQN variable: " + var_name)
            assert(0)  # <3

        return(value)

    def set_value(self, var_name, new_value, tf_session):
        """
        Set the value of the tf variable [var_name] to [new_value]
        """

        if(var_name in self.assign_operator):

            tf_session.run(
                self.assign_operator[var_name], {
                    self.l_param_input[var_name]: new_value})
        else:
            print("Thou shall only assign learning parameters!")
