import time
import torch as t

class BasicModule(t.nn.Module):
	'''
	封装nn.Module， 主要提供save和load两个方法
	'''
	def __init__(self):
		super(BasicModule, self).__init__()
		self.model_name = str(type(self))

	def load(self, path):
		'''
		加载指定路径的模型
		'''
		self.load_state_dict(t.load(path))

	def save(self, name=None):
		'''
		保存模型，默认 “模型名+时间”作为文件名
		'''
		if name is None:
			prefix = 'checkpoints\\' + self.model_name + '_'
			name = time.strftime(prefix + '%m%d_%H%M%S.pth')
		t.save(self.state_dict(), name)
		return name
