import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import time


# 设置torch和numpy生成随机数的随机种子
torch.manual_seed(0)
np.random.seed(0)
# 如果使用了cuda，那么cuda也有随机种子，这里设置为true则得到默认算法，
# 配合随机种子固定，如果输入相同则每次的输出是固定的
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MetaBalance(Optimizer):
	def __init__(self, parameters, relax_factor=0.7, beta=0.9):
		if relax_factor < 0. or relax_factor >= 1.:
			raise ValueError(f'Invalid relax_factor: {relax_factor}, it should be 0. <= relax_factor < 1.')
		if beta < 0. or beta >= 1.:
			raise ValueError(f'Invalid beta: {beta}, it should be 0. <= beta < 1.')
		rel_beta_dict = {'relax_factor': relax_factor, 'beta': beta}
		super(MetaBalance, self).__init__(parameters, rel_beta_dict)

	@torch.no_grad()
	def step(self, losses):
		"""
		Args:
			losses: list, it contains some losses from each auxiliary task and main task
					the first one is main task
		"""
		for idx, loss in enumerate(losses):
			loss.backward(retain_graph=True)
			for group in self.param_groups:
				for gp in group['params']:
					if gp.grad is None:
						print('breaking')
						break
					if gp.grad.is_sparse:
						raise RuntimeError('MetaBalance does not support sparse gradients')
					# store the result of moving average
					state = self.state[gp]
					if len(state) == 0:
						for i in range(len(losses)):
							if i == 0:
								gp.norms = [torch.zeros(1).cuda()]
							else:
								gp.norms.append(torch.zeros(1).cuda())

					beta = group['beta']
					gp.norms[idx] = gp.norms[idx] * beta + (1 - beta) * torch.norm(gp.grad)

					relax_factor = group['relax_factor']
					gp.grad = gp.grad * gp.norms[0] / gp.norms[idx] * relax_factor + 
								gp.grad * (1. - relax_factor)

					if idx == 0:
						state['sum_gradient'] = torch.zero_like(gp.data)
						state['sum_gradient'] += gp.grad
					else:
						state['sum_gradient'] += gp.grad

					if gp.grad is not None:
						gp.grad.detach_()
						gp.grad.zero_()
					if idx == len(losses) - 1:
						gp.grad = state['sum_gradient']

		
