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
from lasagne.updates import get_or_compute_grads
from collections import OrderedDict

from models.cascade_base import CascadeBase

class DigitTrigger(CascadeBase):  
    def __init__(self,
                 img_shape,
                 learning_rate,
                 c,
                 c_complexity,
                 c_sub_objs,
                 c_sub_obj_cs,
                 mul,
                 pool_sizes,
                 num_filters,
                 filter_sizes,
                 optimizer=False,
                 l2_c=0
                ):
        self.img_shape = img_shape
        self.pool_sizes = pool_sizes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.l2_c = l2_c
        
        self.c_sub_objs = theano.shared(np.array(c_sub_objs))
        self.c_sub_obj_cs = theano.shared(np.array(c_sub_obj_cs))
        self.c_complexity = theano.shared(c_complexity)
        self.c = c
        
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        
        assert(len(pool_sizes) == len(num_filters))
        self.num_cascades = len(pool_sizes)
        
        self.out, self.downsampled_activation_layers, self.masked_output_layer = self.build_network()
        
        if mul:
            self.output_layer = self.masked_output_layer
        else:
            self.output_layer = self.out

        assert(len(self.downsampled_activation_layers) == len(c_sub_obj_cs))
        assert(len(self.downsampled_activation_layers) == len(c_sub_objs))
        
        self.output = self.build_output()
        self.target_pool_layers = self.build_target_pool_layers()
        
        self.train = self.compile_trainer(learning_rate, optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    # NOTE(aelphy): batchnorm here should be used smartly due to l2
    def build_network(self):
        net = InputLayer((None, 1) + tuple(self.img_shape),
                         self.input_X,
                         name='network input')
        
        convs = []

        # Build network
        for i in range(self.num_cascades):
            net=Conv2DLayer(net,
                            nonlinearity=elu,
                            num_filters=self.num_filters[i],
                            filter_size=self.filter_sizes[i],
                            pad='same',
                            name='conv {}'.format(i + 1))
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
    
    def optimizer(self,
                  loss_or_grads,
                  params,
                  learning_rate=0.002,
                  beta1=0.9,
                  beta2=0.999,
                  epsilon=1e-8):
        """Adamax updates
            with reset
        """
        all_grads = get_or_compute_grads(loss_or_grads, params)
        t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        t = t_prev + 1
        a_t = learning_rate/(one-beta1**t)
        
        shareds = []

        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            
            shareds.append(m_prev)
            shareds.append(u_prev)

            m_t = beta1*m_prev + (one-beta1)*g_t
            u_t = T.maximum(beta2*u_prev, abs(g_t))
            step = a_t*m_t/(u_t + epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates, shareds
    
    def compile_trainer(self, learning_rate, optimizer):
        obj = self.get_obj()

        params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        
        updates, self.opt_shareds = self.optimizer(obj,
                                                   params,
                                                   learning_rate=learning_rate)
            
        return theano.function([self.input_X, self.targets], 
                               {
                                'obj' : self.get_obj(),
                                'recall' : self.get_recall(),
                                'precision' : self.get_precision(),
                                'accuracy' : self.get_accuracy(),
                                'loss' : self.get_loss(),
                                'sub_loss' : self.get_sub_loss(),
                                'total_complexity' : self.get_total_complexity(),
                                'complexity_parts' : T.stack(self.get_complexity_parts())
                               },
                               updates=updates)
    
    
    def save(self, path, name):
        layers = lasagne.layers.get_all_param_values(self.masked_output_layer)
        np.savez(os.path.join(path, name), *(layers + list(self.c_sub_objs.get_value()) + [self.c_complexity.get_value()] + list(self.c_sub_obj_cs.get_value())))
    
    def load(self, path, name):
        layers = lasagne.layers.get_all_param_values(self.masked_output_layer)
        
        with np.load(os.path.join(path, name + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.masked_output_layer, param_values[:len(layers)])
            self.c_sub_objs.set_value(np.array(param_values[len(layers):len(layers) + len(self.c_sub_objs.get_value())]))
            self.c_complexity.set_value(param_values[len(layers) + len(self.c_sub_objs.get_value())])
            self.c_sub_obj_cs.set_value(np.array(param_values[len(layers) + len(self.c_sub_objs.get_value()) + 1:]))
            