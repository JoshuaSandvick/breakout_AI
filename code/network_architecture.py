import tensorflow as tf

class Q_Network:
    def __init__(self, batch_size):
        # Network size parameters
        num_frames_back_to_convolve = 4
        num_hidden_layer_1_output_channels = 16
        num_hidden_layer_2_output_channels = 32
        num_fc_1_output_units = 256
        num_output_actions = 4 # noop, left, right, fire
    
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        
        # Input layer
        self.input_layer = tf.placeholder(tf.uint8, shape=(None, 84, 84, num_frames_back_to_convolve))
        batch_size = tf.shape(self.input_layer)[0]
        
        # Normalize the frames
        X = tf.to_float(self.input_layer) / 255.0
        # First hidden layer
        filter_1 = tf.Variable(tf.random_normal([8, 8, num_frames_back_to_convolve, num_hidden_layer_1_output_channels], dtype=tf.float32), dtype=tf.float32)
        h1 = tf.nn.conv2d(X, filter_1, padding="VALID", strides=[1, 4, 4, 1])
        self.hidden_layer_1 = tf.nn.relu(h1)
        #self.hidden_layer_1 = tf.nn.max_pool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        
        # Second hidden layer
        filter_2 = tf.Variable(tf.random_normal([4, 4, num_hidden_layer_1_output_channels, num_hidden_layer_2_output_channels], dtype=tf.float32), dtype=tf.float32)
        h2 = tf.nn.conv2d(self.hidden_layer_1, filter_2, padding="VALID", strides=[1, 2, 2, 1])
        self.hidden_layer_2 = tf.nn.relu(h2)
        #self.hidden_layer_2 = tf.nn.max_pool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

        # Flatten the hidden layer's output for the fully connected layer
        #flattened = tf.layers.flatten(self.hidden_layer_2)
        # Need magic # for weird bug w/ None type on -1 input
        flattened = tf.contrib.layers.flatten(self.hidden_layer_2)
        #flattened = tf.reshape(self.hidden_layer_2, [batch_size, 128])
        
        # First fully-connected layer
        self.fc_layer_1 = tf.layers.dense(flattened, units=num_fc_1_output_units, activation=tf.nn.relu)
        
        # Output layer
        self.output_layer = tf.layers.dense(self.fc_layer_1, units=num_output_actions)
        
        # Create loss
        gather_indicies = tf.range(batch_size) * tf.shape(self.output_layer)[1] + self.actions
        self.action_predictions = tf.gather(tf.reshape(self.output_layer, [-1]), gather_indicies)
        
        losses = tf.squared_difference(self.targets, self.action_predictions)
        self.loss = tf.reduce_mean(losses)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
