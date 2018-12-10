import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import torch
import torch.nn.functional as F

from robot_collision import RobotCollision
from maze_collision import MazeCollision


#############################
#       PYTORCH SETUP       #
#############################

print("PyTorch version: ", torch.__version__)
print("CUDA available:  ", torch.cuda.is_available())
print("CUDA version:    ", torch.version.cuda)
v = str(torch.backends.cudnn.version())
print("CUDNN version:   ", "%s.%s.%d"%(v[0], v[1], int(v[2:])))
print("CUDA device:     ", torch.cuda.get_device_name(0))

assert torch.cuda.is_available() # We require a GPU
torch.set_default_tensor_type(torch.cuda.FloatTensor) # by default, use float32 tensors on the GPU
torch.backends.cudnn.benchmark = True
_ = torch.as_tensor(1.0) # GPU initialization happens when the first tensor is loaded



#############################
#       PROBLEM SETUP       #
#############################

# FLAGS
PLOT = True
VERBOSE = False
if VERBOSE:
    print("\nWARNING: VERBOSE is TRUE, due to printing and GPU -> CPU transfers, optimization will be slower!\n")

# CONSTANTS:
GRAVITY = 9.81 # m/s^2

# CONSTRAINTS:
# Warning, changing any of the following will likely require changes to the optimal hyper-parameters
SAMPLING_FREQUENCY = 20.0
THRUST_MAGNITUDE_BOUNDS = [5.0, 15.0] # m/s^2
BODYRATE_MAXIMUM = 30.0 # rad/second
COLLISION_DISTANCE = 0.25 # meters
GOAL_POSITION_RADIUS = 0.05 # meters deviation from the goal position allowed
GOAL_VELOCITY_RADIUS = 0.05 # m/s deviation from the goal velocity allowed

# HYPERPARAMETERS:
LEARNING_RATE = 0.0014633750727427797
W_QUAD_COLLISION = 11620.030281393
W_MAZE_COLLISION = 270.677287006
W_GOAL_POS = 18589.569408167
W_GOAL_VEL = 21134.417963886
W_THRUST = 1e5 # thrust limit never hit so this value is arbitrary
W_BODYRATE = 1e5 # body rate limit never hit so this value is arbitrary




#############################
#        MAZE SETUP         #
#############################

# LOAD MAZE:
MAZEDATA = []
with open('maze.txt', 'r') as f:
    while True:
        line = f.readline()
        if len(line)>0:
            line = [0 if i=='0' else 1 for i in line.strip().replace(' ','')]
            MAZEDATA.append(line)
        else:
            break
# flip and transpose so we can access the maze using indexing as expected
# i.e. (0,0) is the lower left of the text file
MAZEDATA = np.array(MAZEDATA)[::-1,:].T 


class Maze(object):
    def __init__(self, maze, rad):
        from scipy.sparse.csgraph import shortest_path
        X, Y = maze.shape
        self.X = X
        self.Y = Y
        self.map = maze
        self.tensor = torch.as_tensor(maze.copy(), dtype=torch.uint8)
        self.rad = rad
        self.collisions = MazeCollision(self.tensor, rad)
        
        G = np.zeros((4*X*Y, 4*X*Y))
        for x in np.arange(0.5,X,0.5):
            for y in np.arange(0.5,Y,0.5):
                idx = self.indexOf(x,y)
                assert self.coordOf(idx)[0]==x and self.coordOf(idx)[1]==y
                
                if self.collision(x,y):
                    G[idx,:] = 0
                    G[:,idx] = 0
                else:
                    for xd in [-1,-0.5,0,0.5,1]:
                        for yd in [-1,-0.5,0,0.5,1]:
                            if xd==0 and yd==0 or abs(xd)==1 and abs(yd)==1:
                                continue
                            elif x+xd<=0 or x+xd>=self.X or y+yd<=0 or y+yd>=self.Y:
                                continue
                            elif abs(xd)!=1 and abs(yd)!=1 and not self.collision(x+xd,y+yd):
                                G[idx, self.indexOf(x+xd,y+yd)] = np.sqrt(xd**2 + yd**2)
                            elif abs(xd)==1 and abs(yd)!=1 and not self.collision(x+xd,y+yd) and not self.collision(x+0.5*np.sign(xd),y+yd):
                                G[idx, self.indexOf(x+xd,y+yd)] = np.sqrt(xd**2 + yd**2)
                            elif abs(xd)!=1 and abs(yd)==1 and not self.collision(x+xd,y+yd) and not self.collision(x+xd,y+0.5*np.sign(yd)):
                                G[idx, self.indexOf(x+xd,y+yd)] = np.sqrt(xd**2 + yd**2)
                
        # We've now "solved" the maze, computing the minimum cost and shortest path from every point on the graph to every other
        dist, pred = shortest_path(G, directed=False, return_predecessors=True)

        # Possible starting locations in the middle of the maze
        possibleStarts = 0.5+np.array([(x,y) for x in range(3, X-3) for y in range(3,Y-3) if maze[x,y]==1])

        # Possible goal locations on the border
        goals = 0.5+np.array([(x,y) for x in range(X) for y in range(Y) if maze[x,y]==1 and (x<2 or x>=X-2 or y<2 or y>=Y-2)])

        assert len(possibleStarts)>=len(goals)

        self.dist = dist
        self.pred = pred
        self.goals = goals
        self.possibleStarts = possibleStarts
        self.N = len(goals)
    
    def __getitem__(self, xy):
        x,y = xy
        return self.map[x,y]
    
    def collision(self, x, y):
        d = [[0,0], [0,-0.5], [-0.5,0], [-0.5,-0.5], [0, -1], [-1, 0], [-0.5,-1], [-1, -0.5], [-1,-1]]
        if x<=0 or x>=self.X or y<=0 or y>=self.Y:
            return True
        for (dx,dy) in d:
            if x+dx>=0 and y+dy>=0 and x+dx==int(x+dx) and y+dy==int(y+dy) and self.map[int(x+dx), int(y+dy)]==0:
                return True
        return False
    
    def indexOf(self, x, y):
        return int(2*y+2*x*2*self.Y)
    
    def coordOf(self, i):
        x,y = ((i//(2*self.Y))/2, (i-2*self.Y*(i//(2*self.Y)))/2)
        assert i==self.indexOf(x,y)
        return x,y
        
    def distanceFrom(self, p0, p1):
        i0 = self.indexOf(*p0)
        i1 = self.indexOf(*p1)
        return self.dist[i0,i1]
    
    def randomProblem(self):
        starts = np.random.permutation(self.possibleStarts)[:self.N]
        goals = self.optimalGoalsFor(starts)
        return starts, goals
    
    def optimalGoalsFor(self, starts):
        from scipy.optimize import linear_sum_assignment
        N = len(self.goals)
        D = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                D[i, j] = self.distanceFrom(starts[i,:], self.goals[j,:])**2
        _, ci = linear_sum_assignment(D)
        return self.goals[ci,:]
    
    def waypointsFrom(self, p0, p1):
        i = self.indexOf(*p0)
        j = self.indexOf(*p1)
        p = j
        path = [self.coordOf(p)]
        while True:
            p = self.pred[i,p]
            path.append(self.coordOf(p))
            if p==i:
                break
        return np.array(path[::-1])
    
    def splinesFrom(self, p0, p1, transitionTime=10, waitTime=0.5, k=3, s=0):
        path = self.waypointsFrom(p0, p1)

        t = np.zeros(path.shape[0])
        # initialize the waypoint times based on the cumulative % of path-length
        # not perfect but it will do for this example
        t[1:] = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        t = t/t[-1]*transitionTime # scale the transition times

        # now we extend/pad the trajectory with repeated start and end points in order to 
        # constrain the jerk, acceleration, and velocity to roughly 0 at the start and end
        te = np.zeros(len(t)+2*k)
        te[k:-k] = t+waitTime # offset the transition times by the wait times
        te[0:k] = np.arange(0,k)/k*waitTime # divide waitTime by the k padded points at the beginning 
        te[-k:] = np.arange(1,k+1)/k*waitTime+max(t)+waitTime # likewise at the end
        path = np.concatenate([[path[0,:] for _ in range(k)], path, [path[-1,:]  for _ in range(k)]], axis=0)

        return splrep(te, path[:,0], k=k, s=s), splrep(te, path[:,1], k=k, s=s)

        

def PlanTrajectories(starts, goals, maze):
    TS = 1.0/SAMPLING_FREQUENCY

    pos=[]
    for si in range(200):
        s = maze.splinesFrom(starts[si], goals[si])
        tmax = s[0][0][-1] # s[0] is the x spline, s[0][0] is the time of the knots, s[0][0][-1] is the final time
        t = np.arange(0,tmax+TS/2,TS)
        x = splev(t, s[0])+0.5
        y = splev(t, s[1])+0.5
        pos.append(np.array([x, y]).T)
    pos = torch.as_tensor(np.stack(pos, axis=0), dtype=torch.float)
        
    N = pos.shape[1]

    # Matrix MJ = P
    M = TS**3/6 * torch.as_tensor(np.tri(N-1, k=0), dtype=torch.float)
    for i in range(1, N-1):
        M = M + i*TS**3 * torch.as_tensor(np.tri(N-1, k=-i), dtype=torch.float)
    M_POS_FROM_JERK = M

    # Matrix MJ = V
    M = TS**2/2 * torch.as_tensor(np.tri(N-1, k=0), dtype=torch.float)
    for i in range(1, N-1):
        M = M + TS**2 * torch.as_tensor(np.tri(N-1, k=-i), dtype=torch.float)
    M_VEL_FROM_JERK = M

    # Matrix MJ = A
    M_ACC_FROM_JERK = TS*torch.as_tensor(np.tri(N-1, k=0), dtype=torch.float)

    # We now stack the matrices together and solve the linear-least-squares problem.
    # We constrain the entire position trajectory, the final velocity, and regularize on difference from the polynomial jerk trajectory

    A = torch.cat([1e3*M_POS_FROM_JERK,
                   1e5*M_POS_FROM_JERK[-1:,:],
                   1e5*M_VEL_FROM_JERK[-1:,:],
                   1e1*M_ACC_FROM_JERK,
                   1e1*(torch.eye(N-1))], dim=0)

    B = torch.cat([1e3*(pos[:,1:,:]-pos[:,0:1,:]),
                   1e5*(pos[:,-1:,:]-pos[:,0:1:,:]),
                   1e5*torch.zeros((pos.shape[0], 1, pos.shape[2])),
                   1e1*torch.zeros((pos.shape[0], N-1, pos.shape[2])),
                   1e1*torch.zeros((pos.shape[0], N-1, pos.shape[2]))], dim=1)
    J = torch.stack([torch.gels(B[i,:,:], A)[0] for i in range(B.shape[0])], dim=0)[:,:int(N)-1,:]
    A = torch.matmul(M_ACC_FROM_JERK, J)
    P = torch.matmul(M_POS_FROM_JERK, J) + pos[:,0:1,:]
    
    J = J.transpose(0,1).contiguous()
    J = torch.cat([J, torch.zeros(J.shape[0], J.shape[1], 1)], dim=-1) # add the z dimension
    return J





g = torch.as_tensor([[[0, 0, -GRAVITY]]], dtype=torch.float) # define this outside so we're only initializing once

def QuadrocopterDynamicsConstraints(pos, vel, acc, jerk):
    # pos, vel, acc, and jerk are all GPU Tensors with shape: [#time, #vehicles, #axes],
    # e.g. 200 timesteps, 100 quads, 3 axes (x, y, z) would have shape [200, 100, 3]
    
    fmin, fmax = THRUST_MAGNITUDE_BOUNDS
    wmax = BODYRATE_MAXIMUM

    # norm across axes to get the magnitudes for each vehicle at each timestep
    fmag = (acc-g).norm(p=2, dim=-1)
    wmag = jerk.norm(p=2, dim=-1)/fmag

    # ReLU, a common activation function for neural networks, is defined as: ReLU(x) = max(x, 0)
    # We use it here to zero points which do not violate constraints
    floss = F.relu(fmag-fmax) + F.relu(fmin-fmag) # zero if fmax >= fmag >= fmin i.e. the constraint is not violated
    wloss = F.relu(wmag-wmax) # only non-zero if wmag>wmax, i.e. the constraint is violated

    # return a scalar loss (for gradient descent), as well as boolean tensors indicating constraint violation
    # note if loss == 0, all trajectories are feasible
    return floss, wloss, (floss>0).int(), (wloss>0).int()



def GoalConstraints(pos, vel, acc, jerk, goalPosition):
    p_radius_final = (pos[-1,:,:]-goalPosition).norm(p=2, dim=-1)
    v_radius_final = (vel[-1,:,:]-0).norm(p=2, dim=-1) # 0 m/s end velocity

    ploss = p_radius_final**2 
    vloss = v_radius_final**2

    return ploss, vloss, (p_radius_final>GOAL_POSITION_RADIUS).int(), (v_radius_final>GOAL_VELOCITY_RADIUS).int()




def IntegrateJerkTrajectory(jerk, startPosition):
    # jerk is a GPU tensor with shape [#time, #vehicles, #axes]
    # startPosition is a GPU tensor with shape [#vehicle, #axes]
    
    TS = 1.0/SAMPLING_FREQUENCY
    
    acc = TS*torch.cumsum(jerk,dim=0) # a0=0
    ca = TS*torch.cumsum(acc,dim=0)
    vel = ca + TS/2.0 * acc # v = v0 + TS*a + 0.5 TS^2*j, note that acc = TS*cumsum(jerk) and v0=0
    pos = startPosition[None,:,:] + TS*torch.cumsum(vel,dim=0) + TS/2.0 * ca + TS**2/6.0 * acc
    
    return pos, vel, acc 




maze = Maze(MAZEDATA, rad=COLLISION_DISTANCE/2) # this is a one-time operation, so do it before the timing

QuadCollisionConstraint = RobotCollision(COLLISION_DISTANCE, sortDim=0)

def TimedSolve(MAXITER=30000, SEED=None):

    if SEED is not None:
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.manual_seed(SEED)
    else:
        np.random.seed()
        torch.cuda.seed_all()

    # Initialize a random problem
    starts, goals = maze.randomProblem()

    # Now we load the start and goal positions onto the GPU for the rest of the computation
    startPosition = torch.cat([torch.as_tensor(starts, dtype=torch.float) + 1e-3*(torch.rand(starts.shape[0], 2)-0.5), 1e-3*torch.rand(starts.shape[0], 1)], dim=-1)
    goalPosition = torch.cat([torch.as_tensor(goals, dtype=torch.float), torch.zeros(goals.shape[0], 1)], dim=-1)


    input('Random Problem Initialized. Press ENTER to solve!')
    print('Solving... ', end='', flush=True)
    startTime = time.time()

    # Plan the initial trajectories
    jerk = PlanTrajectories(starts, goals, maze)
    jerk.requires_grad=True
    optimizer = torch.optim.Adam([jerk], lr=LEARNING_RATE)

    i=0
    solved = False
    while i<MAXITER:
        # zero back-propogated gradients
        optimizer.zero_grad()
        
        # integrate the jerk trajectory
        pos, vel, acc = IntegrateJerkTrajectory(jerk, startPosition)
        mazeCollisions = maze.collisions(pos)

        # compute the various losses
        floss, wloss, thrustInfeasible, rateInfeasible = QuadrocopterDynamicsConstraints(pos, vel, acc, jerk)
        quadCollisions, closs = QuadCollisionConstraint(pos)
        ploss, vloss, finalPosInfeasible, finalVelInfeasible = GoalConstraints(pos, vel, acc, jerk, goalPosition)
                
        # compute the total loss
        loss =  W_THRUST * floss.sum() + \
                W_BODYRATE * wloss.sum() + \
                W_QUAD_COLLISION * quadCollisions + \
                W_MAZE_COLLISION * mazeCollisions + \
                W_GOAL_POS *  ploss.sum() + \
                W_GOAL_VEL * vloss.sum()
        
        if VERBOSE and i%100==0:
            print("Iteration %d, Violations: Thrust=%d, Rate=%d, QColl=%d, MColl=%d, Final Pos=%d, Final Vel=%d" % (
                i,
                int(thrustInfeasible.sum()),
                int(rateInfeasible.sum()),
                int(quadCollisions),
                int(mazeCollisions),
                int(finalPosInfeasible.sum()),
                int(finalVelInfeasible.sum())
            ), flush=True)
        i+=1 
        
        if torch.all(finalPosInfeasible==0) and torch.all(finalVelInfeasible==0) and torch.all(thrustInfeasible==0) and torch.all(rateInfeasible==0) and quadCollisions==0 and mazeCollisions==0:
            solved = True
            break
        
        loss.backward() # backpropogate gradients
        optimizer.step()


    if solved:
        print("Solved in %.3f seconds (%d iterations)" % (time.time()-startTime, i))
    else:
        print("Unsolved after maximum iterations reached (%.3f seconds, %d iterations)" % (time.time()-startTime, i))
    
    return pos.detach().cpu().numpy() # return, for example, the position trajectory as a numpy array for plotting



if __name__=='__main__':
    print('Setup Complete.')
    while True:
        print('') # TimedSolve will setup a random problem and then solve it
        p = TimedSolve()
        if PLOT:
            plt.figure(figsize=(6,6))
            ax=plt.subplot(111)
            plt.xlim(0,MAZEDATA.shape[0])
            plt.ylim(0,MAZEDATA.shape[1])

            for i in range(MAZEDATA.shape[0]):
                for j in range(MAZEDATA.shape[1]):
                    if MAZEDATA[i,j]==0:
                        ax.add_patch(plt.Rectangle((i,j), 1, 1, edgecolor='k', linewidth=0.1))

            cmap = plt.cm.Set1
            C = [cmap(i) for i in range(cmap.N)]

            for i in range(p.shape[1]):
                plt.plot(p[0,i,0], p[0,i,1], 'o', color=C[i%len(C)], markersize=5)
                plt.plot(p[-1,i,0], p[-1,i,1], '^', color=C[i%len(C)], markersize=5)

            for i in range(p.shape[1]):
                plt.plot(p[:,i,0], p[:,i,1], color=C[i%len(C)], linewidth=1.25, alpha=0.5)

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()