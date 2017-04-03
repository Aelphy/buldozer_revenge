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

class Buldozer(object):
    def __init__(self,
                 cascades_builders,
                 cascades_params,
                 img_shape=(640, 480),
                 learning_rate=1e-3,
                 c=1.0,
                 c_complexity=1e-1,
                 c_sub_objs=[1e-3, 1e-3],
                 c_sub_obj_cs=[1e-3, 1e-3],
                 mul=True,
                 optimizer=lasagne.updates.adamax,
                 l2_c=0,
                 c_obj=1
                ):
        self.img_shape = img_shape
        self.l2_c = l2_c
        self.c_obj = c_obj

        self.c_sub_objs = c_sub_objs
        self.c_sub_obj_cs = c_sub_obj_cs
        self.c_complexity = c_complexity
        self.c = c

        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')

        assert(len(c_sub_objs) == len(c_sub_obj_cs) == len(cascades_params) == len(cascades_builders))
        self.num_cascades = len(cascades_params)

        (self.out,
         self.downsampled_activation_layers,
         self.masked_output_layer,
         self.complexities) = self.build_network(cascades_builders, cascades_params)

        if mul:
            self.output_layer = self.masked_output_layer
        else:
            self.output_layer = self.out

        assert(len(self.downsampled_activation_layers) == len(self.c_sub_obj_cs) == len(self.c_sub_objs))

        self.output = self.build_output()
        self.target_pool_layers = self.build_target_pool_layers()

        self.train = self.compile_trainer(learning_rate, optimizer)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()

    def build_output(self):
        return lasagne.layers.get_output(self.output_layer, self.input_X)

    def compute_loss(self, a, t, c):
        return -(t * T.log(a + 1e-6) + c * (1.0 - t) * T.log(1.0 - a + 1e-6)).mean()

    def get_pool_size(self, smaller_dim, higher_dim):
        return np.array(higher_dim)[-2:] / np.array(smaller_dim)[-2:]

    def build_target_pool_layers(self):
        result = []
        input_layer = InputLayer((None, 1) + tuple(self.img_shape),
                                 self.targets,
                                 name='target transform input layer')
        for i in range(len(self.downsampled_activation_layers)):
            pool_size = self.get_pool_size(lasagne.layers.get_output_shape(self.downsampled_activation_layers[i]),
                                           self.img_shape)
            result.append(MaxPool2DLayer(input_layer, pool_size=pool_size))

        result.append(
                      MaxPool2DLayer(
                                     input_layer,
                                     pool_size=self.get_pool_size(lasagne.layers.get_output_shape(self.output_layer),
                                                                  self.img_shape)
                                    )
                     )

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
        complexity = []
        max_complexity = []
        constants = []

        for i in range(len(self.downsampled_activation_layers) - 1):
            activation_layer = self.downsampled_activation_layers[i]
            targets = lasagne.layers.get_output(self.target_pool_layers[i], self.targets)

            pool_size = self.get_pool_size(lasagne.layers.get_output_shape(self.downsampled_activation_layers[i + 1]),
                                           self.img_shape)
            constants.append(np.prod(pool_size))
            complexity.append((lasagne.layers.get_output(activation_layer) * (1 - targets)).sum())
            max_complexity.append((1 - targets).sum())

        return complexity, max_complexity, self.complexities, constants

    def get_total_complexity(self):
        result, max_result = self.get_total_absolute_complexity()

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

        return result

    def get_loss(self):
        a = self.output.ravel()
        t = lasagne.layers.get_output(self.target_pool_layers[-1], self.targets).ravel()

        return self.compute_loss(a, t, self.c)

    def get_obj(self):
        l2_penalty = 0

        for layer in lasagne.layers.get_all_layers(self.output_layer):
            l2_penalty += regularize_layer_params(layer, l2)

        return self.c_obj * self.get_loss() +\
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

    def build_network(self, cascades_builders, cascades_params):
        net = InputLayer((None, 1) + tuple(self.img_shape),
                         self.input_X,
                         name='network input')

        cascade_outs = []
        complexities = []

        # Build network
        for i in range(self.num_cascades):
            net, complexity = cascades_builders[i](net, *cascades_params[i])
            cascade_outs.append(net)
            complexities.append(complexity)


        out = Conv2DLayer(net,
                          nonlinearity=sigmoid,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          name='prediction layer')

        # NOTE: Early stop classifiers and multiply step
        branches = [None] * self.num_cascades

        # Build branches
        for i in range(self.num_cascades):
            branches[i] = Conv2DLayer(cascade_outs[i],
                                      num_filters=1,
                                      filter_size=1,
                                      nonlinearity=sigmoid,
                                      name='decide network {} output'.format(i + 1))

        downsampled_activation_layers = [branches[0]]

        for i in range(self.num_cascades - 1):
            downsampled_activation_layers.append(
                                                 MaxPoolMultiplyLayer(
                                                   branches[i + 1],
                                                   downsampled_activation_layers[-1],
                                                   self.get_pool_size(
                                                       lasagne.layers.get_output_shape(branches[i + 1]),
                                                       lasagne.layers.get_output_shape(downsampled_activation_layers[-1])
                                                                      )
                                                                     )
                                                )
        masked_out = MaxPoolMultiplyLayer(
                        out,
                        downsampled_activation_layers[-1],
                        self.get_pool_size(lasagne.layers.get_output_shape(out),
                                           lasagne.layers.get_output_shape(downsampled_activation_layers[-1])))

        return out, downsampled_activation_layers, masked_out, complexities

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
                                'obj' : obj,
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
