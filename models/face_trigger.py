import os
import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import Conv2DLayer,\
                           MaxPool2DLayer,\
                           InputLayer
from lasagne.nonlinearities import elu, sigmoid, rectify
from lasagne.regularization import l2, regularize_layer_params
from utils.maxpool_multiply import MaxPoolMultiplyLayer

from models.cascade_base import CascadeBase

class FaceTrigger(CascadeBase):        
    def build_network(self):
        net = lasagne.layers.batch_norm(InputLayer((None, 1) + tuple(self.img_shape),
                                        self.input_X,
                                        name='network input'))
        
        convs = []

        # Build network
        for i in range(self.num_cascades):
            net = lasagne.layers.batch_norm(Conv2DLayer(net,
                                            nonlinearity=elu,
                                            num_filters=self.num_filters[i],
                                            filter_size=self.filter_sizes[i],
                                            pad='same',
                                            name='conv {}'.format(i + 1)))
            convs.append(net)
            net = MaxPool2DLayer(net,
                                 pool_size=self.pool_sizes[i],
                                 name='Max Pool {} {}'.format(i + 1, i + 2))

        
        out = Conv2DLayer(net,
                          nonlinearity=sigmoid,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          name='prediction layer')
        
        branches = [None] * self.num_cascades

        # Build branches
        for i in range(self.num_cascades):
            branches[i] = Conv2DLayer(convs[i],
                                      num_filters=1,
                                      filter_size=1,
                                      nonlinearity=sigmoid,
                                      name='decide network {} output'.format(i + 1))

        downsampled_activation_layers = [branches[0]]

        for i in range(self.num_cascades - 1):
            downsampled_activation_layers.append(MaxPoolMultiplyLayer(branches[i + 1],
                                                                      downsampled_activation_layers[-1],
                                                                      self.pool_sizes[i]))
        masked_out = MaxPoolMultiplyLayer(out,
                                          downsampled_activation_layers[-1],
                                          self.pool_sizes[-1])
        
        return out, downsampled_activation_layers, masked_out