import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F


class EpsilonGreedyActionSelector():
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.argmax_select(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions

    def argmax_select(scores):
    	return np.argmax(scores, axis=1)



class DQNAgent():
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=None):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        #self.preprocessor = preprocessor
        self.device = device

    def initial_state(self):
    	return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


# def default_states_preprocessor(states):
#     """
#     Convert list of states into the form suitable for model. By default we assume Variable
#     :param states: list of numpy arrays with states
#     :return: Variable
#     """
#     if len(states) == 1:
#         np_states = np.expand_dims(states[0], 0)
#     else:
#         np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
#     return torch.tensor(np_states)