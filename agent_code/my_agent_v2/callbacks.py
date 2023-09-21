import os
import pickle
import random
import numpy as np
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT','BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train:
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Qtable = pickle.load(file)
        np.savetxt("Qtable.txt",self.Qtable.reshape(-1))
        self.epsilon=0.1
        print(self.epsilon)


def softmax_action_selection(Q_values, temperature=1.0):
    Q_values[np.isneginf(Q_values)] = -1e9
    Q_values=Q_values/1000
    probabilities = np.exp(Q_values / temperature)
    action_probabilities = probabilities / np.sum(probabilities)
    #print(action_probabilities)
    chosen_action = np.random.choice(ACTIONS, p=action_probabilities)
    return chosen_action


def epsilon_greedy_policy_bkp(self,state):
    if np.random.rand() < self.epsilon:
        self.logger.info("Random action.")
        return np.random.choice(ACTIONS,p=[.25, .25, .25, .25])
    else:
        self.logger.info("Model based action.")
        q_values=self.Qtable[state[0],state[1], :]
        
        if all(q == q_values[0] for q in q_values):
            return np.random.choice(ACTIONS,p=[.25, .25, .25, .25])
        else:
            return ACTIONS[np.argmax(self.Qtable[state[0],state[1], :])]
        
def epsilon_greedy_policy(self,state):
    if np.random.rand() < self.epsilon:
        self.logger.info("Random action.")
        return np.random.choice(ACTIONS,p=[.24, .24, .24, .24,0.04])
    else:
        self.logger.info("Model based action.")
        q_values=self.Qtable[state[0],state[1], :]
        #print(q_values)
        return softmax_action_selection(q_values, temperature=1.0)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    state = state_to_features(game_state)
    action = epsilon_greedy_policy(self,state)
    self.logger.debug(state)
    self.logger.debug(action)
    return action

def situational_awareness(game_state):
    x,y=game_state['self'][3]
    return {'UP':game_state['field'][x][y+1],'RIGHT':game_state['field'][x+1][y],'DOWN':game_state['field'][x][y-1],'LEFT':game_state['field'][x-1][y]}



"""
def coins_in_row(game_state):
    coins=game_state['coins']
    agent_x, agent_y = game_state['self'][3]

    #Check for coins in each directions
    left_coins = any(coin_x < agent_x and coin_y == agent_y for coin_x, coin_y in coins)
    right_coins = any(coin_x > agent_x and coin_y == agent_y for coin_x, coin_y in coins)
    up_coins= any(coin_y < agent_y and coin_x == agent_x for coin_x, coin_y in coins)
    down_coins= any(coin_y < agent_y and coin_x == agent_x for coin_x, coin_y in coins)

    return {'UP':up_coins,'RIGHT': right_coins,'DOWN':down_coins,'LEFT':left_coins}
"""


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.                          

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    #print(game_state['explosion_map'])
    features=[]
    if game_state is None:
        return None
    agent_x, agent_y = game_state['self'][3]
    #width =s.COLS-2
    #state_index = agent_y *width  + agent_x
    features.append(agent_x) 
    features.append(agent_y)
    features.append(situational_awareness(game_state))
    #features.append(coins_in_row(game_state))
    #features.append(find_direction_with_max_coins(game_state))
    #features.append(bomb_trajectory(game_state,escape_path))
    #features.append(find_direction_with_max_bombs(game_state))

    #print(features)
    return features

