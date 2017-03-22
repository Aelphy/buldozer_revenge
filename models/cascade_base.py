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

class CascadeBase(object):
    def __init__(self,
                 img_shape=(640, 480),
                 learning_rate=1e-3,
                 c=1.0,
                 c_complexity=1e-3,
                 c_sub_objs=[1e-3, 1e-3],
                 c_sub_obj_cs=[1e-3, 1e-3],
                 mul=True,
                 pool_sizes=[2, 2, 5],
                 num_filters=[1, 1, 3],
                 filter_sizes=[1, 3, 3],
                 optimizer=lasagne.updates.adamax,
                 l2_c=0
                ):
        self.img_shape = img_shape
        self.pool_sizes = pool_sizes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.l2_c = l2_c
        
        self.c_sub_objs = c_sub_objs
        self.c_sub_obj_cs = c_sub_obj_cs
        self.c_complexity = c_complexity
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

        assert(len(self.downsampled_activation_layers) == len(self.c_sub_obj_cs))
        assert(len(self.downsampled_activation_layers) == len(self.c_sub_objs))
        
        self.output = self.build_output()
        self.target_pool_layers = self.build_target_pool_layers()
        
        self.train = self.compile_trainer(learning_rate, optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def build_output(self):
        return lasagne.layers.get_output(self.output_layer, self.input_X)
    
    def compute_loss(self, a, t, c):
        return -(t * T.log(a + 1e-6) + c * (1.0 - t) * T.log(1.0 - a + 1e-6)).mean()
    
    def build_target_pool_layers(self):
        result = []
        input_layer = InputLayer((None, 1) + tuple(self.img_shape),
                                 self.targets,
                                 name='target transform input layer')
        for i in range(len(self.downsampled_activation_layers)):
            result.append(MaxPool2DLayer(input_layer, pool_size=int(np.prod(self.pool_sizes[:i]))))
            
        result.append(MaxPool2DLayer(input_layer, pool_size=np.prod(self.pool_sizes)))
            
        return result

    def get_sub_loss(self):
        sub_obj = 0
        
        for i, activation_layer in enumerate(self.downsampled_activation_layers):
            sub_answer = lasagne.layers.get_output(activation_layer, self.input_X)
            targets = lasagne.layers.get_output(self.target_pool_layers[i], self.targets)
            
            sub_obj += self.compute_loss(sub_answer.ravel(),
                                         targets.ravel(),
                                         self.c_sub_obj_cs[i]) * self.c_sub_objs[i]
            
        return sub_obj
    
    def compute_complexity(self):
        complexity = [np.prod(self.img_shape)]
        max_complexity = [np.prod(self.img_shape)]
        miltipliers = []
        constants = []
        
        for i in range(len(self.downsampled_activation_layers) - 1):
            activation_layer = self.downsampled_activation_layers[i]
            targets = lasagne.layers.get_output(self.target_pool_layers[i], self.targets)

            constants.append(np.prod(np.array(self.img_shape) / np.prod(self.pool_sizes[:i + 1])))
            miltipliers.append(self.num_filters[i + 1] * self.filter_sizes[i + 1] ** 2)
            complexity.append((lasagne.layers.get_output(activation_layer) * (1 - targets)).sum())
            max_complexity.append((1 - targets).sum())
            
        return complexity, max_complexity, miltipliers, constants
    
    def get_total_complexity(self):
        complexity, max_complexity, miltipliers, constants = self.compute_complexity()
        result = complexity[0]
        max_result = max_complexity[0]
        
        for i in range(len(complexity) - 1):
            result += complexity[i + 1] * miltipliers[i] + constants[i]
            max_result += max_complexity[i + 1] * miltipliers[i] + constants[i]
            
        return result / max_result
    
    def get_total_absolute_complexity(self):
        complexity, max_complexity, miltipliers, constants = self.compute_complexity()
        result = complexity[0]
        max_result = max_complexity[0]
        
        for i in range(len(complexity) - 1):
            result += complexity[i + 1] * miltipliers[i] + constants[i]
            max_result += max_complexity[i + 1] * miltipliers[i] + constants[i]
            
        return result, max_result
    
    def get_complexity_parts(self):
        complexity, max_complexity, miltipliers, constants = self.compute_complexity()
        
        result = []
        
        for i in range(len(complexity)):
            result.append(complexity[i] / max_complexity[i])
            
        result[0] = T.constant(result[0])
        
        return result
    
    def get_loss(self):
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()
        
        return self.compute_loss(a, t, self.c)
    
    def get_obj(self):
        l2_penalty = 0
        
        for layer in lasagne.layers.get_all_layers(self.output_layer):
            l2_penalty += regularize_layer_params(layer, l2)
        
        return self.get_loss() +\
               self.get_sub_loss() +\
               self.c_complexity * self.get_total_complexity() +\
               self.l2_c * l2_penalty
    
    def get_precision(self):   
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()
        
        return ((a > 0.5) * t).sum() / (a > 0.5).sum()
    
    def get_recall(self):   
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()
        
        return ((a > 0.5) * t).sum() / t.sum()
    
    def get_accuracy(self):      
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()
        
        return lasagne.objectives.binary_accuracy(a, t).mean()

    def build_network(self):
        net = InputLayer((None, 1) + tuple(self.img_shape),
                         self.input_X,
                         name='network input')
        
        convs = []

        # Build network
        for i in range(self.num_cascades):
            net = Conv2DLayer(net,
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
        
        self.branches = branches # TODO(aelphy): move to constructor
        
        return out, downsampled_activation_layers, masked_out

    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)

    def compile_evaluator(self):        
        return theano.function([self.input_X, self.targets], {
                                                              'obj' : self.get_obj(),
                                                              'recall' : self.get_recall(),
                                                              'precision' : self.get_precision(),
                                                              'accuracy' : self.get_accuracy(),
                                                              'loss' : self.get_loss(),
                                                              'sub_loss' : self.get_sub_loss(),
                                                              'total_complexity' : self.get_total_complexity(),
                                                              'complexity_parts' : T.stack(self.get_complexity_parts())
                                                             })
    
    def compile_trainer(self, learning_rate, optimizer):
        obj = self.get_obj()

        params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        
        updates = optimizer(obj,
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
        np.savez(os.path.join(path, name), *lasagne.layers.get_all_param_values(self.masked_output_layer))
    
    def load(self, path, name):
        with np.load(os.path.join(path, name + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.masked_output_layer, param_values)
