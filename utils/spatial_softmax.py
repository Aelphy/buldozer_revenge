from lasagne.nonlinearities import *
from lasagne.layers import Layer
import theano
import theano.tensor as T

class SpatialSoftmaxLayer(Layer):
    """
    Softmax layer that computes the softmax over pixels in the same location,
    i.e., over the channel axis. This layer will automatically use the CuDNN
    version of this softmax if it is available.
    
    Parameters
    ----------
    
    incoming : a :class:`Layer`
    dnn_softmax_mode : if CuDNN is enabled, what mode should we use for
        that implementation. There are two: 'accurate', and 'fast'
    """
    def __init__(self, incoming, dnn_softmax_mode='accurate', **kwargs):
        super(SpatialSoftmaxLayer, self).__init__(incoming, **kwargs)
        self.input_shape = incoming.output_shape

    def get_output_for(self, inpt, **kwargs):
        return T.exp(inpt) / T.exp(inpt).sum(axis=1).reshape([-1, 1] + list(self.input_shape[-2:]))