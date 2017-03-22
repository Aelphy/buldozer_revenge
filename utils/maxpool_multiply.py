
import theano.tensor as T
from lasagne.layers import MaxPool2DLayer, ElemwiseMergeLayer

def MaxPoolMultiplyLayer(activation_layer_small, activation_layer_big, pool_size, name=None):
    max_pool_layer = MaxPool2DLayer(activation_layer_big, pool_size=pool_size)
    return ElemwiseMergeLayer([max_pool_layer,activation_layer_small],T.mul)
