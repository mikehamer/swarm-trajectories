import os.path
mypath = os.path.split(__file__)[0]

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as torchcpp

MazeCollisionCpp = torchcpp.load(name='MazeCollision', sources=[os.path.join(mypath, 'maze_collision.cpp'), os.path.join(mypath, 'maze_collision.cu')])

class MazeCollisionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, maze, rad):
        count, grad = MazeCollisionCpp.forward(pos, maze, rad)
        ctx.save_for_backward(count, grad)
        ctx.rad = rad
        return count

    @staticmethod
    def backward(ctx, dL_dC):
        count, grad = ctx.saved_tensors
        if count==0:
            return torch.zeros_like(grad), None, None
        else:
            return grad*dL_dC, None, None

class MazeCollision(torch.nn.Module):
    def __init__(self, maze, rad):
        super(MazeCollision, self).__init__()
        self.maze = maze
        self.rad = rad
    
    def forward(self, pos):
        return MazeCollisionAutogradFunction.apply(pos, self.maze, self.rad)