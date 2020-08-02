import os
from tensorboardX import SummaryWriter
from model_retinaNet import resnet18

writer = SummaryWriter(logdir=logdir, flush_secs=2)
alpha = 0.25
gamma = 2.0
model = resnet18(num_classes = 3, alpha, gamma)
model.eval() # not training
model.cuda() # on gpu
model.calculate_focalLoss = False
dummy_input = torch.rand(1,3,224,224).cuda() # random input on gpu
'''
very important to note that 
1) the input must be tensor
2) write can only keep track of tensor, not native python objects.
this becomes problematic when the model transforms anchor boxes
'''
                    
write.add_graph(model, input_to_model=dummy_input,
                verbose=True)
writer.close()