import tensorflow as tf
from tensorflow.keras import layers

class GraphConvolutionLayer(layers.Layer):
    def __init__(self, unit, activation=None, dropout_rate=0., edges=5, name='', **kwargs):
        super(GraphConvolutionLayer, self).__init__(name=name, **kwargs)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dense1 = [layers.Dense(units=unit) for _ in range(edges-1)]
        self.dense2 = layers.Dense(units=unit)
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation(activation)
        
    def call(self, inputs, training=False):
        ic = len(inputs)
        if ic<2:
            raise ValueError('GraphConvolutionLayer requires at leat two inputs: [adjacency_tensor, node_tensor]')
            
        adjacency_tensor = inputs[0]
        node_tensor = inputs[1]
        
        #means that this is second loop
        if ic > 2:
            hidden_tensor = inputs[2]
            annotations = tf.concat((hidden_tensor, node_tensor), -1)
        else:
            annotations = node_tensor
            
        output = tf.stack([dense(annotations) for dense in self.dense1], 1)  
        
        adj = tf.transpose(adjacency_tensor[:, :, :, 1:], (0, 3, 1, 2))  
        
        output = tf.matmul(adj, output)      
        output  = tf.reduce_sum(output, 1) + self.dense2(node_tensor)
        output = self.activation(output)
        output = self.dropout(output)
        return adjacency_tensor, node_tensor, output

class GraphAggregationLayer(layers.Layer):
    def __init__(self, units, activation=None, dropout_rate=0., name='', **kwargs):
        super(GraphAggregationLayer, self).__init__(name=name, **kwargs)
        self.d1 = layers.Dense(units = units, activation='sigmoid')
        self.d2 = layers.Dense(units = units, activation=activation)
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation(activation)
        
    def call(self, inputs, training=False):
        i = self.d1(inputs)
        j = self.d2(inputs)
        output = tf.reduce_sum(i*j,1)
        output = self.activation(output)
        output = self.dropout(output)
        return output
    
class MultiGraphConvolutionLayer(layers.Layer):
    def __init__(self, units, activation=None, dropout_rate = 0., edges=5, name='', **kwargs):
        super(MultiGraphConvolutionLayer, self).__init__(name=name, **kwargs)
        
        if len(units) < 2:
            raise ValueError('Single layer unit provided, this layer is for multiple convolutions only. Use GraphConvolutionLayer instead.') 
        
        self.first_convolution =  GraphConvolutionLayer(units[0], activation, dropout_rate, edges)
        self.gcl = [GraphConvolutionLayer(u, activation, dropout_rate, edges) for u in units[1:]]

        
    def call(self, inputs, training=False):
        adjacency_tensor = inputs[0]
        node_tensor = inputs[1]
        
        tensors = self.first_convolution([adjacency_tensor, node_tensor])
        
        for layer in self.gcl:
            tensors = layer(tensors)
        
        _,_, hidden_tensor = tensors
            
        return hidden_tensor
    
class GraphEncoderLayer(layers.Layer):
    def __init__(self, units, activation='tanh', dropout_rate=0., edges= 5, name='', **kwargs):
        super(GraphEncoderLayer, self).__init__(name=name, **kwargs)
        graph_convolution_units, auxiliary_units = units
        
        self.multi_graph_convolution_layer =  MultiGraphConvolutionLayer(graph_convolution_units, activation, dropout_rate, edges)
        self.graph_aggregation_layer = GraphAggregationLayer(auxiliary_units, activation, dropout_rate)
        
    def call(self, inputs, training=False):
        output = self.multi_graph_convolution_layer(inputs)
        
        node_tensor= inputs[1]
        
        if len(inputs)>2:
            hidden_tensor = inputs[2]
            annotations = tf.concat((output, hidden_tensor, node_tensor),-1)
        else:
            _,node_tensor = inputs                    
            annotations = tf.concat((output,node_tensor),-1)
        
        output = self.graph_aggregation_layer(annotations)
        return output
    
