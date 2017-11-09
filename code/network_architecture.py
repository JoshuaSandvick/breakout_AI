import tensorflow as tf

def create_network(batch_size):
    # Network size parameters
    num_frames_back_to_convolve = 4
    num_hidden_layer_1_output_channels = 16
    num_hidden_layer_2_output_channels = 32
    num_fc_1_output_units = 256
    num_output_actions = 4 # noop, left, right, fire

    target = tf.placeholder(shape=[None], dtype=tf.float32)
    
    # Input layer
    input_layer = tf.placeholder(tf.float16, shape=(batch_size, 84, 84, num_frames_back_to_convolve))
    
    # First hidden layer
    filter_1 = tf.Variable(tf.random_normal([8, 8, num_frames_back_to_convolve, num_hidden_layer_1_output_channels], dtype=tf.float16), dtype=tf.float16)
    hidden_layer_1 = tf.nn.conv2d(input_layer, filter_1, padding="VALID", strides=[1, 4, 4, 1])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    hidden_layer_1 = tf.nn.max_pool(hidden_layer_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
    
    # Second hidden layer
    filter_2 = tf.Variable(tf.random_normal([4, 4, num_hidden_layer_1_output_channels, num_hidden_layer_2_output_channels], dtype=tf.float16), dtype=tf.float16)
    hidden_layer_2 = tf.nn.conv2d(hidden_layer_1, filter_2, padding="VALID", strides=[1, 2, 2, 1])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    hidden_layer_2 = tf.nn.max_pool(hidden_layer_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
    
    # Flatten the hidden layer's output for the fully connected layer
    flattened = tf.reshape(hidden_layer_2, [batch_size, -1])
    
    # First fully-connected layer
    fc_layer_1 = tf.layers.dense(flattened, units=num_fc_1_output_units, activation=tf.nn.relu)
    
    # Output layer
    output_layer = tf.layers.dense(fc_layer_1, units=num_output_actions)
    
    return output_layer