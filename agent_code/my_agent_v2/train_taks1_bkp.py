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
gamma = 0.9  # Discount factor
initial_epsilon = 0.9  # Initial epsilon value

min_initial_value = 0.01
max_initial_value = 0.1
min_epsilon = 0.5     # Minimum epsilon value
epsilon_decay_rate = 0.1  # Amount by which epsilon is reduced after each episode



#Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))

#TRANSITION_HISTORY_SIZE = 3  
#RECORD_ENEMY_TRANSITIONS = 1.0  

#New Events
SCORE_INCREASE="SCORE_INCREASE"
REVISIT_STATE="REVISIT_STATE"
COINS_HAUL_20="COINS_HAUL_20"
COINS_HAUL_50="COINS_HAUL_50"
LOW_COINS_HAUL="LOW_COINS_HAUL"
AVG_COINS_HAUL="AVG_COINS_HAUL"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    width = s.COLS
    height = s.ROWS
    #num_states = width * height
    #initial_value=0
    num_actions=len(ACTIONS)
    if not os.path.isfile("my-saved-model.pt"):
        #self.Qtable=np.random.uniform(min_initial_value, max_initial_value, size=(width,height, num_actions))
        self.Qtable=np.zeros((width,height, num_actions))
        #self.Qtable=np.full((width,height, num_actions),initial_value)
    else:
        print("Loading existing Qtable")
        with open("my-saved-model.pt", "rb") as file:
            self.Qtable = pickle.load(file)
    self.reward=0
    self.epsilon=initial_epsilon
    self.visited_states=[]
    with open("rewards.txt", "a") as file:
        file.write("\n New Epsilon:"+str(self.epsilon) + "\n")


def q_learning_update(self,state, action, reward, next_state):
    current_q = self.Qtable[state[0],state[1], action]
    if current_q==-np.inf:
        pass
    else:
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
    if old_game_state['step']==1 and old_game_state['round']==1:
        x,y = np.where(old_game_state['field']== -1)
        self.Qtable[x,y, :]= -np.inf
    if old_game_state['step']==1:
        print(" New Round",new_game_state['round'],"\nReward",self.reward,"\nPosition",old_game_state['self'][3])
    old_score=old_game_state['self'][1]
    new_score=new_game_state['self'][1]
    if new_score>old_score:
        events.append(SCORE_INCREASE)
    if new_game_state['step']>1:
        if state_to_features(new_game_state)[:2]==self.visited_states[-1]:
            events.append(REVISIT_STATE)
    self.visited_states.append(state_to_features(old_game_state)[:2])
    self.reward+=reward_from_events(self,events,old_game_state)
    #with open("run_log.txt", "a") as file:
        #file.write("--------------------------------------\n")
        #file.write(str(self.Qtable[state_to_features(old_game_state)[0],state_to_features(old_game_state)[1]]))
        #file.write(str(self_action)+str(ACTIONS.index(self_action))+str(self.reward))
    q_learning_update(self,state_to_features(old_game_state), ACTIONS.index(self_action), self.reward,state_to_features(new_game_state))
    #with open("run_log.txt", "a") as file:
        #file.write(str(self.Qtable[state_to_features(old_game_state)[0],state_to_features(old_game_state)[1]]))
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events,new_game_state)))
   

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
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events,last_game_state)))
    self.reward+=reward_from_events(self,events,last_game_state)
    q_learning_update(self,state_to_features(last_game_state), ACTIONS.index(last_action), self.reward,state_to_features(last_game_state))
    print("Reward:", self.reward)
    print("Score:", last_game_state['self'][1])
    with open("rewards.txt", "a") as file:
        file.write(str(self.reward) + ",")
    if last_game_state['round']%1000==0:
        self.epsilon=max(self.epsilon - epsilon_decay_rate, min_epsilon)
        with open("rewards.txt", "a") as file:
            file.write("\n New Epsilon:"+str(self.epsilon) + "\n")
    self.reward=0
    self.visited_states=[]
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Qtable, file)


def reward_from_events(self, events: List[str],game_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    features=state_to_features(game_state)[2]
    #coins=state_to_features(game_state)[3]
    max_coins=state_to_features(game_state)[3]
    action_event_mapping={
        'UP':e.MOVED_UP, 'DOWN':e.MOVED_DOWN,'LEFT':e.MOVED_LEFT, 'RIGHT':e.MOVED_RIGHT
    }
    game_rewards = {
        e.INVALID_ACTION:-1,
        e.MOVED_LEFT:0,
        e.MOVED_RIGHT:0,
        e.MOVED_UP:0,
        e.MOVED_DOWN:0,
        e.COIN_COLLECTED:4,
        SCORE_INCREASE:4,
        REVISIT_STATE:-1 
        ,COINS_HAUL_20:10,
        COINS_HAUL_50:15,
        LOW_COINS_HAUL:-5,
        AVG_COINS_HAUL:2.5
    }
    for action, value in features.items():
        if value == -1:
            game_rewards[action_event_mapping[action]] =0
        elif value == 0:
            game_rewards[action_event_mapping[action]] = 1
            if action==max_coins:
                game_rewards[action_event_mapping[action]] += 1
    
    if e.SURVIVED_ROUND in events:
        score=game_state['self'][1]
        if score>=20:
            events.append(COINS_HAUL_20)
        elif score==50:
            events.append(COINS_HAUL_50)
        elif score<10:
            events.append(LOW_COINS_HAUL)
        else:
            events.append(AVG_COINS_HAUL)
    #print(game_rewards)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum





