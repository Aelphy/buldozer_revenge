from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

class VGG():
    def __init__(self, 
                 input_shape,
                 learning_rate=1e-3,
                 optimizer=lasagne.updates.adamax):
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        
    def perform_compilation(self, num_cascades):
        self.network = self.build_model(input_shape, num_cascades)
        self.output = lasagne.layers.get_output(self.network)
        self.train = self.compile_trainer(learning_rate, optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def get_precision(self):
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()

        return ((a > 0.5) * t).sum() / (a > 0.5).sum()

    def get_recall(self):
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()

        return ((a > 0.5) * t).sum() / t.sum()
    
    def get_total_complexity(self):
        complexity = 0
        
        #for layer in lasagne.layers
            
        
        
    def compile_evaluator(self):
        return theano.function([self.input_X],
                               {
                                'obj' : self.get_obj(),
                                'recall' : self.get_recall(),
                                'precision' : self.get_precision(),
                                'accuracy' : self.get_accuracy(),
                                'loss' : self.get_loss(),
                                'sub_loss' : self.get_sub_loss(),
                                'total_complexity' : self.get_total_complexity()
                               },
                               updates=updates)
    
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
                                'loss' : self.get_loss(),
                                'sub_loss' : self.get_sub_loss(),
                                'total_complexity' : self.get_total_complexity()
                               },
                               updates=updates)

    
    def build_model(self, input_shape, num_cascades):
        net = InputLayer([None] + input_shape)
        
        for i in range(num_cascades):
            net = ConvLayer(net, 2**(6 + i), 3, pad=1, flip_filters=False)
            net = ConvLayer(net, 2**(6 + i), 3, pad=1, flip_filters=False)
            net = PoolLayer(net, 2)

        return net