import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict

class GraphConvolutionLayer(layers.Layer):
    """ Graph convolution layer used in molecular GANs. Based on work by Nicola De Cao et al. https://arxiv.org/abs/1805.11973"""
    def __init__(self, units, activation=None, dropout_rate=0., edges=5, name='', **kwargs):
        super(GraphConvolutionLayer, self).__init__(name=name, **kwargs)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.units = units
        self.edges = edges
        
        self.dense1 = [layers.Dense(units=self.units) for _ in range(edges-1)]
        self.dense2 = layers.Dense(units=self.units)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.activation = layers.Activation(self.activation)
        
    def call(self, inputs, training=False):
        ic = len(inputs)
        if ic<2:
            raise ValueError('GraphConvolutionLayer requires at leat two inputs: [adjacency_tensor, node_features_tensor]')
            
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
    
    def get_config(self) -> Dict:
        """Returns config dictionary for this layer."""
        config = super(GraphConvolutionLayer, self).get_config()
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        config['units'] = self.units
        config['edges'] = self.edges
        return config

class GraphAggregationLayer(layers.Layer):
    """ Graph Aggregation layer used in molecular GANs. Based on work by Nicola De Cao et al. https://arxiv.org/abs/1805.11973"""
    def __init__(self, units, activation=None, dropout_rate=0., name='', **kwargs):
        super(GraphAggregationLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.d1 = layers.Dense(units = units, activation='sigmoid')
        self.d2 = layers.Dense(units = units, activation=activation)
        self.dropout_layer = layers.Dropout(dropout_rate)
        self.activation_layer = layers.Activation(activation)
        
    def call(self, inputs, training=False):
        i = self.d1(inputs)
        j = self.d2(inputs)
        output = tf.reduce_sum(i*j,1)
        output = self.activation_layer(output)
        output = self.dropout_layer(output)
        return output
    
    def get_config(self) -> Dict:
        """Returns config dictionary for this layer."""
        config = super(GraphAggregationLayer, self).get_config()
        config['units'] = self.units
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        config['edges'] = self.edges
        return config
    
class MultiGraphConvolutionLayer(layers.Layer):
    """Multiple calls on Graph convolution layer where data from previous layer is used in the next one."""
    def __init__(self, units, activation=None, dropout_rate = 0., edges=5, name='', **kwargs):
        super(MultiGraphConvolutionLayer, self).__init__(name=name, **kwargs)
        if len(units) < 2:
            raise ValueError('Single layer unit provided, this layer is for multiple convolutions only. Use GraphConvolutionLayer instead.') 
                
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.edges = edges
        
        self.first_convolution =  GraphConvolutionLayer(self.units[0], self.activation, self.dropout_rate, self.edges)
        self.gcl = [GraphConvolutionLayer(u, self.activation, self.dropout_rate, self.edges) for u in self.units[1:]]

        
    def call(self, inputs, training=False):
        adjacency_tensor = inputs[0]
        node_tensor = inputs[1]
        
        tensors = self.first_convolution([adjacency_tensor, node_tensor])
        
        for layer in self.gcl:
            tensors = layer(tensors)
        
        _,_, hidden_tensor = tensors
            
        return hidden_tensor
    
    def get_config(self) -> Dict:
        """Returns config dictionary for this layer."""
        config = super(MultiGraphConvolutionLayer, self).get_config()
        config['units'] = self.units
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        config['edges'] = self.edges
        return config
    
class GraphEncoderLayer(layers.Layer):
    """ Graph encoder layer used in molecular GANs. Based on work by Nicola De Cao et al. https://arxiv.org/abs/1805.11973"""
    def __init__(self, units, activation='tanh', dropout_rate=0., edges= 5, name='', **kwargs):
        super(GraphEncoderLayer, self).__init__(name=name, **kwargs)
        self.graph_convolution_units, self.auxiliary_units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.edges = edges
        
        self.multi_graph_convolution_layer =  MultiGraphConvolutionLayer(self.graph_convolution_units, self.activation, self.dropout_rate, self.edges)
        self.graph_aggregation_layer = GraphAggregationLayer(self.auxiliary_units, self.activation, self.dropout_rate)
        
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
    
    def get_config(self) -> Dict:
        """Returns config dictionary for this layer."""
        config = super(GraphEncoderLayer, self).get_config()
        config['graph_convolution_units'] = self.graph_convolution_units
        config['auxiliary_units'] = self.auxiliary_units
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        config['edges'] = self.edges
        return config
    
