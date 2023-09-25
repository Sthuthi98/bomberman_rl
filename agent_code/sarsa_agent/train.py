import numpy as np
import pickle
from typing import List
from collections import deque
import events as e
from .callbacks import state_to_features,select_action


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

alpha = 0.1 # Learning rate
gamma = 0.4  # Discount factor

#Custom Events
VALID_ACTION="VALID_ACTION"
IN_BLAST_RADIUS="IN_BLAST_RADIUS"
TOWARDS_TARGET="TOWARDS_TARGET"
SCORE_INCREASE="SCORE_INCREASE"
IN_LOOP="IN_LOOP"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug('Successfully entered training setup code')
   
    self.initial_position=0
    self.reward=0
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0  
    self.epsilon=0.4  
    self.tau=1.5
    self.transaction_history=deque([], 5)

    
    

def update_sarsa_q_value(self, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Update the Q-value using the SARSA (State-Action-Reward-State-Action) update rule.

    Args:
        Q (dict): The Q-table representing state-action values. It's a dictionary where keys are states,
                  and values are dictionaries of action-value pairs.
        state (hashable): The current state.
        action (hashable): The action taken in the current state.
        reward (float): The immediate reward received after taking the action.
        next_state (hashable): The next state after taking the action.
        next_action (hashable): The action selected in the next state.
        alpha (float): The learning rate (0 to 1) controlling the update step size.
        gamma (float): The discount factor (0 to 1) for future rewards.
    """
    # Retrieve the current Q-value for the state-action pair
    current_q_value = self.Qtable[state][action]
    
    # Retrieve the Q-value for the next state-action pair
    if next_state is None or next_action is None:
        next_q_value=0
    else:
        next_q_value = self.Qtable[next_state][next_action]
    
    # Calculate the updated Q-value using the SARSA update rule
    updated_q_value = current_q_value + alpha * (reward + gamma * next_q_value - current_q_value)
    
    # Update the Q-value in the Q-table
    self.Qtable[state][action] = updated_q_value



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
    action_towards_target=[]
    in_blast_radius=[]
    if old_game_state['step']==1:
        self.initial_position=old_game_state['self'][3]
    #print(self.initial_position)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    channels=state_to_features(old_game_state)
    state=channels[0]
    possible_actions = [action for action, is_possible in zip(ACTIONS[:4], channels[1]) if is_possible == 0]
    if channels[2] is not None:
        (bombx,bomby),t=channels[2][0],channels[2][1]
        if t>0:
            in_blast_radius.append((bombx,bomby))
    action_towards_target.append(channels[3])
    #print(state)
    #print(possible_actions)
    #print(action_towards_target)
    #print(in_blast_radius)
    if self_action in possible_actions:
        events.append(VALID_ACTION)
    if self_action in action_towards_target:
        events.append(TOWARDS_TARGET)
    if state in in_blast_radius:
        events.append(IN_BLAST_RADIUS)
    old_score=old_game_state['self'][1]
    new_score=new_game_state['self'][1]
    if new_score>old_score:
        events.append(SCORE_INCREASE)
    self.reward+=reward_from_events(self,events)
    next_state=state_to_features(new_game_state)
    ideal_next_action=select_action(self,next_state[0])
    self.transaction_history.append(state)
    if self.transaction_history.count(state) > 2:
        events.append(IN_LOOP)

    update_sarsa_q_value(self, state, self_action, reward_from_events(self,events), next_state[0], ideal_next_action, alpha, gamma)


    


    


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
    self.reward+=reward_from_events(self,events)
    state=state_to_features(last_game_state)[0]
    update_sarsa_q_value(self, state, last_action, reward_from_events(self,events), None, None, alpha, gamma)

    #LOgging reward and score
    #print("Cumulative Reward:", self.reward)
    #print("Final Score:", last_game_state['self'][1])
    # Store the model
    if self.initial_position==(1,1):
        with open("my-saved-model-1.pt", "wb") as file:
            pickle.dump(self.Qtable, file)
    if self.initial_position==(1,15):
        with open("my-saved-model-2.pt", "wb") as file:
            pickle.dump(self.Qtable, file)
    if self.initial_position==(15,1):
        with open("my-saved-model-3.pt", "wb") as file:
            pickle.dump(self.Qtable, file)
    if self.initial_position==(15,15):
        with open("my-saved-model-4.pt", "wb") as file:
            pickle.dump(self.Qtable, file)
    self.reward=0
    self.coordinate_history = deque([], 5)
        


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT:0,
        e.MOVED_RIGHT:0,
        e.MOVED_UP:0,
        e.MOVED_DOWN:0,
        VALID_ACTION:0.75,
        e.INVALID_ACTION:-1,
        e.COIN_FOUND:3,
        e.COIN_COLLECTED: 4,
        e.KILLED_OPPONENT: 5,
        SCORE_INCREASE:2,
        e.BOMB_DROPPED:0,
        e.BOMB_EXPLODED:0,
        e.GOT_KILLED:-2,
        e.KILLED_SELF:-2,
        e.CRATE_DESTROYED:0.25,
        e.SURVIVED_ROUND:10,
        e.WAITED:0,TOWARDS_TARGET:1,
        IN_BLAST_RADIUS:-0.5,IN_LOOP:-0.5
    }
    if e.COIN_FOUND in events:
        game_rewards.update({e.BOMB_DROPPED:-0.25})
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
