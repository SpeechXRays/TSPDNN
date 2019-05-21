require "torch"
require "nn"
require "dpnn"
require 'cunn'
local model = torch.load(arg[1])

b = model.modules[1]
c = nn.SpatialConvolutionMM(3,64,7,7,2,2,3,3)
c.weight = b.weight
c.gradWeight = b.gradWeight
c.bias = b.bias
c.gradBias = b.gradBias
model:remove(1)
model:insert(c,1)
a = model.modules[6]
b = nn.SpatialConvolutionMM(64, 64,1,1)
b.weight = a.weight
b.gradWeight = a.gradWeight 
b.bias = a.bias
b.gradBias = a.gradBias
model:remove(6)
model:insert(b,6)
a = model.modules[9]
b = nn.SpatialConvolutionMM(64, 192, 3,3,1,1,1,1)
b.weight = a.weight 
b.bias = a.bias
model:remove(9)
model:insert(b,9) 
torch.save(arg[2],  model)
