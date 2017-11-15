import tensorflow as tf
import numpy as np
import random
import network_architecture as na
import gym

""" Set up network architecture """
# Training parameters
batch_size = 32
num_epochs = 1000
num_examples = 100000
epsilon = .15

net = na.Q_Network(batch_size)

""" Train the network """
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

""" Set up the OpenAI env"""
env = gym.make('Breakout-v0') 
sp = StateProcessor()

def epsilon_greedy_policy(session, state, epsilon):
    """With probability epsilon select a random action, else 
    perform the best action from the q net.

    Args:
        session: tf.Session.
        state: An (84,84,1) matrix representing the greyscale gamestate.
        epsilon: Probability to select a random action.

    Returns:
        int in [0,1,2,3]. The action for the agent to take.
    """ 
    if np.random.random_sample() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = session.run([net], feed_dict=state)
        action = np.argmax(q_values)
    return action

# Initialize replay memory D to capacity N (=100)
state = env.reset()
replay_memory = []
replay_memory_size = 1000
replay_memory_init_size = 100

state = sp.process_state(session, state)
state = np.stack([state]*4, axis=2)
for i in range(replay_memory_size):
    # TODO should probably vary epsilon as we get better target values 
    action = epsilon_greedy_policy(session, state, epsilon)
    next_state, reward, done, _ = env.step(action) 
    next_state = sp.process(next_state)
    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
    replay_memory.append((state, action, reward, next_state, done))
    if done:
        state = env.reset()
        state = process_state(state)
        state = np.stack([state]*4, axis=2)
    else:
        state = next_state
    
for i in range(0, num_epochs):
    state = env.reset()
    state = sp.process_state(session, state)
    state = np.stack([state]*4, axis=2) # phi fn
    for j in range(0, num_examples, batch_size): 
        # Perform an action in emulator
        action = epsilon_greedy_policy(session, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = sp.process(session, next_state)
        next_state = np.append(state[:,:,1:] np.expand_dims(next_state, 2), axis=2) # phi

        # Update replay memory
        if len(replay_memory) > replay_memory_size:
            replay_memory_size.pop(0)
        replay_memory.append((state, action, reward, next_state, done))

        # Sample random minibatch from replay memory 
        minibatch = random.sample(replay_memory, batch_size)
        states, action, reward, next_states, done = map(np.array, *minibatch)

        # Compute the q values and targets
        q_values_next = session.run([net.output_layer], {net.input_layer:next_states})
        targets = reward + np.invert(done).astype(np.float32) * .99 * np.amax(q_values_next, axis=1) 

        # Compute loss and perform gradient descent update
        loss = update(session, states, action, target)
    print("Epoch {}: loss={}".format(i, loss)

