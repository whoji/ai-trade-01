import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SimpleFFDQN(nn.Module):
	"""docstring for SimpleFFDQN"""
	def __init__(self, obs_len, actions_n):
		super(SimpleFFDQN, self).__init__()

		# V: Value
		self.fc_val = nn.Sequential(
			nn.Linear(obs_len, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
		)

		# A: Advantage
		self.fc_adv = nn.Sequential(
			nn.Linear(obs_len, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, actions_n),
		)

	def forward(self, x):
		val = self.fc_val(x)
		adv = self.fc_adv(x)
		return val + adv - adv.mean()


class DQNConv1D(nn.Module):
	"""docstring for SimpleFFDQN"""
	def __init__(self, shape, actions_n):
		super(DQNConv1D, self).__init__()

		# V: Value
		self.conv = nn.Sequential(
			nn.Conv1d(shape[0], 128, 5),
			nn.ReLU(),
			nn.Conv1d(128, 128, 5),
			nn.ReLU(),
		)

		out_size = self._get_conv_out(shape)

		# V: Value
		self.fc_val = nn.Sequential(
			nn.Linear(out_size, 512),
			nn.ReLU(),
			#nn.Linear(512, 512),
			#nn.ReLU(),
			nn.Linear(512, 1),
		)

		# A: Advantage
		self.fc_adv = nn.Sequential(
			nn.Linear(obs_len, 512),
			nn.ReLU(),
			#nn.Linear(512, 512),
			#nn.ReLU(),
			nn.Linear(512, actions_n),
		)

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		conv_out = self.conv(x).view(x.size()[0], -1)
		val = self.fc_val(conv_out)
		adv = self.fc_adv(conv_out)
		return val + adv - adv.mean()

# borrowed from 
# https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)