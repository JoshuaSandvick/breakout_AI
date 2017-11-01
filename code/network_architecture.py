import tensorflow as tf

def create_network(batch_size):
    # Network size parameters
    num_frames_back_to_convolve = 4
    num_hidden_layer_1_output_channels = 16
    num_hidden_layer_2_output_channels = 32
    num_fc_1_output_units = 256
    num_output_actions = 2
    
    # Input layer
    input_layer = tf.placeholder(tf.float16, shape=(batch_size, 105, 80, num_frames_back_to_convolve))
    
    # First hidden layer
    hidden_layer_1 = tf.nn.convolution(input_layer, [8, 8, num_frames_back_to_convolve, num_hidden_layer_1_output_channels], padding="VALID", strides=[4, 4])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    
    # Second hidden layer
    hidden_layer_2 = tf.nn.convolution(hidden_layer_1, [4, 4, num_hidden_layer_1_output_channels, num_hidden_layer_2_output_channels], padding="VALID", stries=[2, 2])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    
    # First fully-connected layer
    fc_layer_1 = tf.layer.dense(hidden_layer_2, units=num_fc_1_output_units, activation=tf.nn.relu)
    
    # Output layer
    output_layer = tf.layer.dense(fc_layer_1, units=num_output_actions)
    
    return output_layer