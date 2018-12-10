import os.path
mypath = os.path.split(__file__)[0]

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as torchcpp

RobotCollisionCpp = torchcpp.load(name='RobotCollision', sources=[os.path.join(mypath, 'robot_collision.cpp'), os.path.join(mypath, 'robot_collision.cu')])

class RobotCollisionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, rad, sortDim):
        _,sortedIdx = torch.sort(pos[:,:,sortDim], dim=1)
        count, loss, grad = RobotCollisionCpp.forward(sortedIdx, pos, rad, sortDim)
        ctx.save_for_backward(count, grad)
        return count, loss

    @staticmethod
    def backward(ctx, dL_dC, *args):
        count, grad = ctx.saved_tensors
        if count==0:
            return torch.zeros_like(grad), None, None
        else:
            dL_dP = grad*dL_dC
            return dL_dP, None, None # return gradient for each of the inputs into forward (#1 pos, #2 rad, #3 sortDim)

class RobotCollision(torch.nn.Module):
    def __init__(self, rad, sortDim=0):
        super(RobotCollision, self).__init__()
        self.rad = rad
        self.sortDim = sortDim
    
    def forward(self, pos):
        return RobotCollisionAutogradFunction.apply(pos, self.rad, self.sortDim)