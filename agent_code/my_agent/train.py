from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import events as e
import settings as s
from .callbacks import state_to_features
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

alpha = 0.1 # Learning rate
gamma = 0.95  # Discount factor
min_initial_value = 0.01
max_initial_value = 0.1
initial_epsilon = 0.9  # Initial epsilon value
min_epsilon = 0.1     # Minimum epsilon value
epsilon_decay_rate = 0.1  # Amount by which epsilon is reduced after each episode



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 3  
#RECORD_ENEMY_TRANSITIONS = 1.0  

#New Events
SCORE_INCREASE="SCORE_INCREASE"
REVISIT_STATE="REVISIT_STATE"
COINS_HAUL_20="COINS_HAUL_20"
COINS_HAUL_50="COINS_HAUL_50"
NO_COINS_HAUL="NO_COINS_HAUL"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    width = s.COLS-1
    height = s.ROWS-1
    #num_states = width * height 
    num_actions=len(ACTIONS)
    if not os.path.isfile("my-saved-model.pt"):
        self.Qtable=np.random.uniform(min_initial_value, max_initial_value, size=(width,height, num_actions))
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.Qtable = pickle.load(file)
    self.reward=0
    self.epsilon=initial_epsilon
    self.visited_states=[]
    with open("rewards.txt", "a") as file:
        file.write("\n New Epsilon:"+str(self.epsilon) + "\n")


def q_learning_update(self,state, action, reward, next_state):
    current_q = self.Qtable[state[0],state[1], action]
    max_next_q = np.max(self.Qtable[next_state[0],next_state[1], :])
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    self.Qtable[state[0],state[1], action] = new_q

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    if new_game_state['step']==1:
        print(" New Round",new_game_state['round'],"\nReward",self.reward)
    self.visited_states.append(state_to_features(old_game_state))
    old_score=old_game_state['self'][1]
    new_score=new_game_state['self'][1]
    if new_score>old_score:
        events.append(SCORE_INCREASE)
    if new_score>=20:
        events.append(COINS_HAUL_20)
    if new_score==50:
        events.append(COINS_HAUL_50)
    if state_to_features(new_game_state) in self.visited_states:
        events.append(REVISIT_STATE)
    if new_game_state['step']==400 and new_score<=20:
        events.append(NO_COINS_HAUL)
    self.reward+=reward_from_events(self,events,new_game_state)
    q_learning_update(self,state_to_features(old_game_state), ACTIONS.index(self_action), self.reward,state_to_features(new_game_state))
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events,new_game_state)))
   

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events,last_game_state)))
    print("Reward:", self.reward)
    print("Score:", last_game_state['self'][1])
    with open("rewards.txt", "a") as file:
        file.write(str(self.reward) + ",")
    if last_game_state['round']%1000==0:
        self.epsilon=max(self.epsilon - epsilon_decay_rate, min_epsilon)
        with open("rewards.txt", "a") as file:
            file.write("\n New Epsilon:"+str(self.epsilon) + "\n")
    self.reward=0
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Qtable, file)


def reward_from_events(self, events: List[str],game_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    agent_x, agent_y = game_state['self'][3]
    coins_haul={
        COINS_HAUL_20:20,
        COINS_HAUL_50:50,
        NO_COINS_HAUL:-10
    }
    if (agent_x,agent_y)==(1,1):
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:-1,
        e.MOVED_RIGHT:2,
        e.MOVED_UP:2,
        e.MOVED_DOWN:-1,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif (agent_x,agent_y)==(15,1):
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:2,
        e.MOVED_RIGHT:-1,
        e.MOVED_UP:2,
        e.MOVED_DOWN:-1 ,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif (agent_x,agent_y)==(1,15):
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:-1,
        e.MOVED_RIGHT:2,
        e.MOVED_UP:-1,
        e.MOVED_DOWN:2,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-1
        }
    elif (agent_x,agent_y)==(15,15):
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:2,
        e.MOVED_RIGHT:-1,
        e.MOVED_UP:-1,
        e.MOVED_DOWN:2,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif agent_x<agent_y and agent_y==15:
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:1,
        e.MOVED_RIGHT:1,
        e.MOVED_UP:-1,
        e.MOVED_DOWN:2,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif agent_x>agent_y and agent_y==1:
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:1,
        e.MOVED_RIGHT:1,
        e.MOVED_UP:2,
        e.MOVED_DOWN:-1,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif agent_x>agent_y and agent_x==15:
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:2,
        e.MOVED_RIGHT:-1,
        e.MOVED_UP:1,
        e.MOVED_DOWN:1,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    elif agent_x<agent_y and agent_x==1:
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:-1,
        e.MOVED_RIGHT:2,
        e.MOVED_UP:1,
        e.MOVED_DOWN:1,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    else: 
        game_rewards = {
        e.INVALID_ACTION:-3,
        e.MOVED_LEFT:1,
        e.MOVED_RIGHT:1,
        e.MOVED_UP:1,
        e.MOVED_DOWN:1 ,
        e.COIN_COLLECTED:20,
        SCORE_INCREASE:15,REVISIT_STATE:-0.5
        }
    game_rewards.update(coins_haul)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum





