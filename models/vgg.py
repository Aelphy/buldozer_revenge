from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, MaxPool2DLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
import lasagne
import theano
import theano.tensor as T
import numpy as np
import os

from lasagne.regularization import l2, l1, regularize_layer_params

class VGG():
    def __init__(self, 
                 input_shape,
                 learning_rate=1e-3,
                 optimizer=lasagne.updates.adamax):
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        
    def perform_compilation(self, num_cascades):
        self.network, self.total_complexity, self.target_output = self.build_model(self.input_shape, num_cascades)
        self.output = lasagne.layers.get_output(self.network, self.input_X)
        self.train = self.compile_trainer(self.learning_rate, self.optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def get_precision(self):
        a = self.output.ravel()
        t = self.target_output.ravel()

        return ((a > 0.5) * t).sum() / ((a > 0.5).sum())

    def get_recall(self):
        a = self.output.ravel()
        t = self.target_output.ravel()

        return ((a > 0.5) * t).sum() / t.sum()
    
    def get_accuracy(self):
        a = self.output.ravel()
        t = self.target_output.ravel()

        return lasagne.objectives.binary_accuracy(a, t).mean()
    
    def get_total_complexity(self):
        return self.total_complexity            
        
    def get_obj(self):
        a = self.output.ravel()
        t = self.target_output.ravel()
        
        return lasagne.objectives.binary_crossentropy(a, t).mean()
        
    def compile_evaluator(self):
        return theano.function([self.input_X, self.targets],
                               {
                                'obj' : self.get_obj(),
                                'recall' : self.get_recall(),
                                'precision' : self.get_precision(),
                                'accuracy' : self.get_accuracy(),
                                'total_complexity' : self.get_total_complexity()
                               })
    
    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)
        
    def compile_trainer(self, learning_rate, optimizer):
        obj = self.get_obj()

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        updates = optimizer(obj,
                            params,
                            learning_rate=learning_rate)

        return theano.function([self.input_X, self.targets],
                               {
                                'obj' : obj,
                                'recall' : self.get_recall(),
                                'precision' : self.get_precision(),
                                'accuracy' : self.get_accuracy(),
                                'total_complexity' : self.get_total_complexity()
                               },
                               updates=updates)

    
    def build_model(self, input_shape, num_cascades):
        input_layer = InputLayer([None] + input_shape)
        net = input_layer
        complexity = 0
        size = np.prod(input_shape[-2:])
        
        for i in range(num_cascades):
            net = ConvLayer(net, num_filters=2**(i + 2), filter_size=3, nonlinearity=lasagne.nonlinearities.tanh, pad='same')
            complexity += size * np.prod(net.filter_size) * net.num_filters
            net = PoolLayer(net, 2)
            
            size = np.prod(lasagne.layers.get_output_shape(net)[-2:])
            
        net = ConvLayer(net, num_filters=1, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid, pad='same')
        complexity += size
            
        target_layer = MaxPool2DLayer(input_layer, pool_size=2**num_cascades)

        return net, theano.shared(complexity), lasagne.layers.get_output(target_layer, self.targets)
    
    def save(self, path, name):
        np.savez(os.path.join(path, name), *lasagne.layers.get_all_param_values(self.network))

    def load(self, path, name):
        with np.load(os.path.join(path, name + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
            