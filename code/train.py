import tensorflow as tf
import numpy as np
import random
import network_architecture as na

""" Set up network architecture """
# Training parameters
batch_size = 32
num_epochs = 1000
num_examples = 100000

net = na.create_network(batch_size)

""" Train the network """
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

example_order = np.linspace(0, num_examples)
for i in range(0, num_epochs):
    example_order = random.shuffle(example_order)
    for j in range(0, num_examples, batch_size):
        batch_order = example_order[j, j+batch_size]
        feed_dict = {input_layer:inputs[:, :, batch_order]}
        
        outputs = session.run([net], feed_dict=feed_dict)       