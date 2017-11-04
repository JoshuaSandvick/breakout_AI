"""Module copied from DennyBritz/reinforcement-learning for preprocessing 
Atari 2600 images for DQN, slightly modified to take advantage of our GPU
convolution implementation.
"""

import tensorflow as tf

class StateProcessor():
    """
    Processes raw Atari images. Resizes it and converts to greyscale.
    """

    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [105, 80], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object.
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [105, 80, 1] state representing grayscale values
        """
        return sess.run(self.output, {self.input_state : state})
