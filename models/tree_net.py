import os
import lasagne
import theano
import theano.tensor as T
import numpy as np
import networkx as nx
from lasagne.layers import Conv2DLayer,\
                           MaxPool2DLayer,\
                           InputLayer,\
                           Upscale2DLayer,\
                           ConcatLayer,\
                           batch_norm
from lasagne.layers import get_output, get_all_params, set_all_param_values
from lasagne.nonlinearities import elu, softmax
from lasagne.regularization import l2, regularize_layer_params
from lasagne.init import GlorotUniform
from collections import OrderedDict
from utils.maxpool_multiply import MaxPoolMultiplyLayer
from utils.spatial_softmax import SpatialSoftmaxLayer as SpatialSoftmax

class TreeNet(object):
    def __init__(self,
                 num_cascades,
                 img_shape,
                 learning_rate=1e-3,
                 optimizer = lasagne.updates.adadelta
                ):
        self.num_cascades = num_cascades
        self.img_shape = img_shape
        
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')

        self.net = self.build_network(num_output_classes=10)

        self.output = self.build_output()

        self.train = self.compile_trainer(learning_rate, optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def apply_4d_shape(self, output):
        eval_params = {self.input_X : np.zeros([1, 1] + list(self.img_shape),
                                               dtype=np.float32)}
        shape = list(output.shape.eval(eval_params))
        
        if len(shape) < 4:
            return output.reshape([-1, 1] + shape[-2:])
            
        return output
        
    def upscale(self, small_output, big_output):
        eval_params = {self.input_X : np.zeros([1, 1] + list(self.img_shape),
                                               dtype=np.float32)}
        secure_small_output = self.apply_4d_shape(small_output)
        small_shape = secure_small_output.shape.eval(eval_params)[1:]
        big_shape = big_output.shape.eval(eval_params)[1:]
        
        pool_size = self.get_pool_size(small_shape, big_shape)
        input_layer = InputLayer([None] + list(small_shape), secure_small_output)
        upscale_layer = Upscale2DLayer(input_layer, pool_size)
        return lasagne.layers.get_output(upscale_layer)

    def upscale_and_add(self, small_output, big_output):
        return self.upscale(small_output, big_output) + self.apply_4d_shape(big_output)
        
    def upscale_and_mul(self, small_output, big_output):   
        return self.upscale(small_output, big_output) * self.apply_4d_shape(big_output)

    def build_output(self):
        result_classes = self.upscale_and_mul(get_output(self.net['classifier_0']),
                                              get_output(self.net['o0'])[:, 0, :, :])
        result_nothing = lasagne.layers.get_output(self.net['o0'])[:, 1, :, :]
                
        for i in range(1, self.num_cascades):
            decision = lasagne.layers.get_output(self.net['o{}'.format(i - 1)])
            
            p_classify = decision[:, 0, :, :]
            p_stop = decision[:, 1, :, :]
            p_continue = decision[:, 2, :, :]
            
            for j in range(i - 1):
                new_p_continue = get_output(self.net['o{}'.format(j)])[:, 2, :, :]
                p_continue = self.upscale_and_mul(new_p_continue, p_continue)
                    
            p_local_stop = self.upscale_and_mul(p_continue,
                                                get_output(self.net['o{}'.format(i)])[:, 1, :, :])
            p_stop = self.upscale_and_add(
                p_stop,
                p_local_stop
            )
            
            p_class = self.upscale_and_mul(get_output(self.net['classifier_{}'.format(i)]),
                                           get_output(self.net['o{}'.format(i)])[:, 0, :, :])
            
            p_local_class = self.upscale_and_mul(p_continue, p_class)
            result_classes = self.upscale_and_add(result_classes,
                                                  p_local_class)
            result_nothing = self.upscale_and_add(result_nothing,
                                                  p_stop)

        preready_result = T.join(1, result_classes, result_nothing)
        input_l = InputLayer([None, 11, 64, 64], preready_result)
        return get_output(SpatialSoftmax(input_l))

    def get_pool_size(self, smaller_dim, higher_dim):
        return np.array(higher_dim)[-2:] / np.array(smaller_dim)[-2:]

    def build_network(self,
                      num_output_classes=10,
                      pad='same',
                      nonlinearity=elu):
        net = OrderedDict()
        net['input'] = InputLayer((None, 1, self.img_shape[0], self.img_shape[1]), self.input_X)

        for i in range(self.num_cascades):
            net['pool_{}'.format(i)] = MaxPool2DLayer(net['input'], 2**i, name='pool_{}'.format(i))


        current = None
        for i in range(self.num_cascades):
            scale_number = self.num_cascades - i - 1

            if current:
                net['i{}'.format(i)] = ConcatLayer([current, net['pool_{}'.format(scale_number)]],
                                                   name='concar_{}'.format(i))
            else:
                net['i{}'.format(i)] = net['pool_{}'.format(scale_number)]

            net['c{}_0'.format(i)] = batch_norm(Conv2DLayer(net['i{}'.format(i)],
                                                            num_filters=16,
                                                            filter_size=3,
                                                            nonlinearity=nonlinearity,
                                                            pad=pad,
                                                            name='c{}_0'.format(i)))

            net['mp{}_0'.format(i)] = MaxPool2DLayer(net['c{}_0'.format(i)], 2, name='mp{}_0'.format(i))
            net['c{}_1'.format(i)] = batch_norm(Conv2DLayer(net['mp{}_0'.format(i)],
                                                            num_filters=32,
                                                            filter_size=3,
                                                            nonlinearity=nonlinearity,
                                                            pad=pad,
                                                            name='c{}_1'.format(i)))
            net['mp{}_1'.format(i)] = MaxPool2DLayer(net['c{}_1'.format(i)], 2, name='mp{}_1'.format(i))

            net['o{}'.format(i)] = SpatialSoftmax(Conv2DLayer(net['mp{}_1'.format(i)],
                                                              num_filters=3,
                                                              filter_size=1,
                                                              nonlinearity=None,
                                                              pad=pad,
                                                              name='o{}'.format(i))
                                                 )
            current = Upscale2DLayer(net['mp{}_1'.format(i)], 8, name='mp{}_1'.format(i))

        num_filters = 64
        W1 = theano.shared(GlorotUniform()((num_filters, 32, 3, 3)))
        b1 = theano.shared(GlorotUniform()((num_filters, 1))).ravel()
        W2 = theano.shared(GlorotUniform()((num_output_classes, num_filters, 3, 3)))
        b2 = theano.shared(GlorotUniform()((num_output_classes, 1))).ravel()
        for i in range(self.num_cascades):
            net['branch_{}_1'.format(i)] = Conv2DLayer(net['mp{}_1'.format(i)],
                                                       num_filters=num_filters,
                                                       filter_size=3,
                                                       nonlinearity=elu,
                                                       pad=pad,
                                                       name='classifier_{}'.format(i),
                                                       W=W1,
                                                       b=b1)
            net['branch_{}_mp'.format(i)] = MaxPool2DLayer(net['branch_{}_1'.format(i)], 2)
            net['classifier_{}'.format(i)] = SpatialSoftmax(Conv2DLayer(net['branch_{}_mp'.format(i)],
                                                                        num_filters=num_output_classes,
                                                                        filter_size=3,
                                                                        nonlinearity=None,
                                                                        pad=pad,
                                                                        name='classifier_{}'.format(i),
                                                                        W=W2,
                                                                        b=b2))

        return net

    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)
    
    def get_accuracy(self):
        return (T.isclose(T.argmax(self.targets, axis=1), T.argmax(self.output, axis=1))).mean()
                                                  
    def get_obj(self):
        return -((self.targets * T.log(self.output)).sum(axis=1)).mean()

    def compile_evaluator(self):
        return theano.function([self.input_X, self.targets], {'obj' : self.get_obj(),
                                                              'accuracy' : self.get_accuracy()})

    def compile_trainer(self, learning_rate, optimizer):
        obj = self.get_obj()

        params = get_all_params(self.net['classifier_{}'.format(self.num_cascades - 1)],
                                trainable=True)

        updates = optimizer(obj,
                            params,
                            learning_rate=learning_rate)

        return theano.function([self.input_X, self.targets],
                               {'obj' : obj, 'accuracy' : self.get_accuracy()},
                               updates=updates)

    def save(self, path, name):
        np.savez(os.path.join(path, name),
                 *get_all_param_values(self.net['classifier_{}'.format(self.num_cascades - 1)]))

    def load(self, path, name):
        with np.load(os.path.join(path, name + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            set_all_param_values(self.net['classifier_{}'.format(self.num_cascades - 1)],
                                 param_values)
