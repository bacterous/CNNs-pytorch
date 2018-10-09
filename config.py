# coding:utf8
import warnings


class DefaultConfig(object):
	env = 'default'
	model = 'ResNet34'

	train_data_root = '\\data\\train'
	test_data_root = '\\data\\test'
	load_model_path = 'checkpoints\\model.pth'

	num_class = 7
	batch_size = 128
	use_gpu = True
	num_workers = 4  # how many workers for loading data
	print_freq = 20  # print info every N batch

	result_file = 'result.csv'

	max_epoch = 10
	lr = 0.1
	lr_decay = 0.95
	weight_decay = 1e-4


def parse(self, kwargs):
	for k, v in kwargs.items():
		if not hasattr(self, k):
			warnings.warn("Warning: opt has no attribute %s %k")
		setattr(self, k, v)

	print('user config:')
	for k, v in self.__class__.__dict__.items():
		print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
