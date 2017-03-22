import lasagne
import numpy as np
import theano.tensor as T
from lasagne.layers import Conv2DLayer,\
                           MaxPool2DLayer,\
                           InputLayer
from lasagne.nonlinearities import elu, sigmoid, rectify
from utils.maxpool_multiply import MaxPoolMultiplyLayer
from models.cascade_base import CascadeBase

class DownsamplingTrigger(CascadeBase):
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
        self.pool_sizes = [2] + pool_sizes # TODO: 2 is just for the check
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
        self.num_cascades = len(pool_sizes) - 1
        
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
        
    def build_target_pool_layers(self):
        result = []
        input_layer = InputLayer((None, 1) + tuple(self.img_shape),
                                 self.targets,
                                 name='target transform input layer')
        for i, activation_layer in enumerate(self.downsampled_activation_layers):
            result.append(MaxPool2DLayer(input_layer, pool_size=np.prod(self.pool_sizes[:i + 2])))
            
        result.append(MaxPool2DLayer(input_layer, pool_size=np.prod(self.pool_sizes)))
            
        return result

    def build_network(self):
        net = InputLayer((None, 1) + tuple(self.img_shape),
                         self.input_X,
                         name='network input')
        net = MaxPool2DLayer(net, pool_size=self.pool_sizes[0])

        # Build network
        for i in range(self.num_cascades + 1):
            net = Conv2DLayer(net,
                              nonlinearity=elu,
                              num_filters=self.num_filters[i],
                              filter_size=self.filter_sizes[i],
                              pad='same',
                              name='conv {}'.format(i + 1))
            net = MaxPool2DLayer(net,
                                 pool_size=self.pool_sizes[i + 1],
                                 name='Max Pool {} {}'.format(i + 1, i + 2))

        
        out = Conv2DLayer(net,
                          nonlinearity=sigmoid,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          name='prediction layer')
        
        branches = [None] * self.num_cascades
        
        layers = lasagne.layers.get_all_layers(out)
        
        # Build branches
        for i in range(self.num_cascades):
            branches[i] = Conv2DLayer(layers[(i + 1) * 2 + 1],
                                      num_filters=1,
                                      filter_size=1,
                                      nonlinearity=sigmoid,
                                      name='decide network {} output'.format(i + 1))

        downsampled_activation_layers = [branches[0]]

        for i in range(self.num_cascades - 1):
            downsampled_activation_layers.append(MaxPoolMultiplyLayer(branches[i + 1],
                                                                      branches[i],
                                                                      self.pool_sizes[i + 2]))
            
        masked_out = MaxPoolMultiplyLayer(out,
                                          downsampled_activation_layers[-1],
                                          self.pool_sizes[-1])
        
        return out, downsampled_activation_layers, masked_out
