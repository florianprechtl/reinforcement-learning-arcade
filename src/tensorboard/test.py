#from tensorboardX import SummaryWriter
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for i in range(20):
	writer.add_scalar('test_scalar', (i+1)*10, i)

writer.close()
