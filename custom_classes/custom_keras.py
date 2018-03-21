from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.layers import Embedding
from keras import *
from keras.layers import Concatenate

class PartiallyTrainableEmbedding(Embedding):
    #untrainable + trainable has to match input size
    def __init__(self, input_dim, untrainable_dim, trainable_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(Embedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.untrainable_dim = untrainable_dim
        self.trainable_dim = trainable_dim
    

    def build(self, input_shape):
        self.trainable_embeddings = self.add_weight(
            shape=(self.trainable_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='trainable_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype, trainable=True)

        self.untrainable_embeddings = self.add_weight(
            shape=(self.untrainable_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='untrainable_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype, trainable=False)
         
        self.built = True
        
        
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
        if regularizer is not None:
            self.add_loss(regularizer(trainable_weight))
            
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
                    
        return weight

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.concatenate([self.untrainable_embeddings, self.trainable_embeddings], axis=0)
        out = K.gather(embeddings, inputs)
        return out
    