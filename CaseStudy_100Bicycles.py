import sys
import os.path
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from robot_collision import RobotCollision


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
def NP(t):
    return t.detach().cpu().numpy()



#############################
#       PROBLEM SETUP       #
#############################

# FLAGS
PLOT = True
VERBOSE = False
if VERBOSE:
    print("\nWARNING: VERBOSE is TRUE, due to printing and GPU -> CPU transfers, optimization will be slower!\n")

# CONSTANTS:
BIKE_REAR_AXLE_DISTANCE = 0.5 # lr
BIKE_FRONT_AXLE_DISTANCE = 0.5 # lf

# CONSTRAINTS:
# Warning, changing any of the following will likely require changes to the optimal hyper-parameters
SAMPLING_FREQUENCY = 20.0
ACCELERATION_MAXIMUM = 2.0 # rad/second
STEER_MAX_ROBOTS_1_TO_50 = np.radians(20)
STEER_MAX_ROBOTS_51_TO_100 = np.radians(70)
COLLISION_DISTANCE = 1.0 # meters
GOAL_POSITION_RADIUS = 0.05 # meters deviation from the goal position allowed
GOAL_VELOCITY_RADIUS = 0.05 # m/s deviation from the goal velocity allowed

# HYPERPARAMETERS:
LEARNING_RATE = 6.94446487e-03
W_ROBOT_COLLISION = 2.14439471e+03
W_GOAL_POS = 2.07196581e+04
W_GOAL_VEL = 5.95288310e+03



#############################
#        LOGO SETUP         #
#############################

LOGOS = {}
for filename in os.listdir('Logos'):
    basename = os.path.splitext(filename)[0].upper()
    filepath = os.path.join('Logos', filename)
    data = np.loadtxt(filepath, delimiter=', ')
    LOGOS[basename] = data[:,[0,2]]*1.65 # scale logos to a 40m x 40m space


def AllocateStartsToGoals(startPosition, unallocatedGoalPosition, random=True):
    if random:
        return np.random.permutation(unallocatedGoalPosition) # randomly shuffle the goal positions
    else:
        from scipy.optimize import linear_sum_assignment
        N = startPosition.shape[0]
        D = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                d = np.sqrt(np.sum((startPosition[i,:]-unallocatedGoalPosition[j,:])**2))
                D[i, j] = -d**2 if d<20 else 9e10
        _, ci = linear_sum_assignment(D)
        return unallocatedGoalPosition[ci,:]



def PlanTrajectories(startPosition, goalPosition):
    # we assume that bikes are facing their goals, ie steer angle = 0, beta0 = 0, phi0 = atan2(dy/dx)
    TS = 1.0/SAMPLING_FREQUENCY
    AM = ACCELERATION_MAXIMUM
    N = startPosition.shape[0]

    dP = goalPosition-startPosition
    phi0 = torch.atan2(dP[:,1], dP[:,0])
    dist = dP.pow(2).sum(dim=-1).sqrt()

    # the number of timesteps it takes for each bike to reach its goal
    T = torch.ceil(torch.sqrt(dist/(0.95*AM))*(10**0.5)/(3**0.25)/TS/2.0)*2
    TM = int(torch.max(T))
    T = TM*TS
    bodyAcc = torch.zeros(TM, N)
    tMask = torch.zeros(TM, N)

    for i in range(N):
        t = torch.as_tensor(np.arange(0, T-TS/2, TS), dtype=torch.float)
        bodyAcc[:TM, i] = 60*dist[i]/T**3*t - 180*dist[i]/T**4*t**2 + 120*dist[i]/T**5*t**3
        tMask[:TM, i] = 1
    
    bodyAccBase = 0.5*torch.log((1+bodyAcc/AM)/(1-bodyAcc/AM)) # arctanh
    steerAngleBase = torch.zeros(TM, N)

    return bodyAccBase, steerAngleBase, phi0, tMask



def GoalConstraints(pos, vel, goalPosition):
    p_radius_final = (pos[-1,:,:]-goalPosition).norm(p=2, dim=-1)
    v_radius_final = (vel[-1,:,:]).norm(p=2, dim=-1) # 0 m/s end velocity

    ploss = p_radius_final**2
    vloss = v_radius_final**2

    return ploss, vloss, (p_radius_final>GOAL_POSITION_RADIUS).int(), (v_radius_final>GOAL_VELOCITY_RADIUS).int()

CollisionConstraints = RobotCollision(COLLISION_DISTANCE, sortDim=0)


STEER_MAX = torch.cat([torch.ones(1,50)*STEER_MAX_ROBOTS_1_TO_50, torch.ones(1,50)*STEER_MAX_ROBOTS_51_TO_100], dim=-1)
def IntegratePlanningTrajectory(bodyAccBase, steerAngleBase, phi0, startPosition):
    TS = 1.0/SAMPLING_FREQUENCY
    lr = BIKE_REAR_AXLE_DISTANCE
    lf = BIKE_FRONT_AXLE_DISTANCE

    steerAngle = STEER_MAX * torch.tanh(steerAngleBase) # |steerAngle| < steer-max
    bodyAcc = ACCELERATION_MAXIMUM * torch.tanh(bodyAccBase)

    bodyVel = TS*torch.cumsum(bodyAcc, dim=0) + 0 # starting body vel is 0
    beta = torch.atan(lr/(lr+lf)*torch.tan(steerAngle))
    phiDot = bodyVel/lr * torch.sin(beta)
    phi = TS*torch.cumsum(phiDot, dim=0) + phi0

    xdot = bodyVel * torch.cos(phi + beta)
    ydot = bodyVel * torch.sin(phi + beta)

    vel = torch.stack([xdot,ydot], dim=-1)
    pos = startPosition[None, :, :] + TS*torch.cumsum(vel, dim=0)

    return pos, vel, phi, beta, bodyVel, steerAngle, bodyAcc




def TimedSolve(start=None, goal=None, MAXITER=30000, SEED=None):

    if SEED is not None:
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.manual_seed(SEED)
    else:
        np.random.seed()
        torch.cuda.seed_all()

    # Define the start positions
    startPosition = np.random.permutation(LOGOS[start or np.random.choice(list(LOGOS.keys()))])
    unallocatedGoalPosition = LOGOS[goal or np.random.choice(list(LOGOS.keys()))] + np.random.normal(scale=1e-5,size=startPosition.shape)

    # Allocate the start positions to goal positions.
    goalPosition = AllocateStartsToGoals(startPosition, unallocatedGoalPosition)

    # Now we load the start and goal positions onto the GPU for the rest of the computation
    startPosition = torch.as_tensor(startPosition, dtype=torch.float)
    goalPosition = torch.as_tensor(goalPosition, dtype=torch.float)

    input('Random Problem Initialized. Press ENTER to solve!')
    print('Solving... ', end='', flush=True)
    startTime = time.time()
    # Plan the initial trajectories
    bodyAccBase, steerAngleBase, phi0, tMask = PlanTrajectories(startPosition, goalPosition)

    # Now setup the optimization rountines
    bodyAccBase.requires_grad = True
    steerAngleBase.requires_grad = True
    optimizer = torch.optim.Adam([bodyAccBase, steerAngleBase], lr=LEARNING_RATE)
    

    
    i = 0
    solved = False
    while i < MAXITER:
        # zero back-propogated gradients
        optimizer.zero_grad()
        
        # integrate the jerk trajectory
        pos, vel, phi, beta, bodyVel, steerAngle, bodyAcc = IntegratePlanningTrajectory(bodyAccBase, steerAngleBase, phi0, startPosition)

        # compute the various losses
        quadCollisions, closs = CollisionConstraints(pos)
        ploss, vloss, finalPosInfeasible, finalVelInfeasible = GoalConstraints(pos, vel, goalPosition)
        treg = ploss.sum()

        # compute the total loss
        loss = W_ROBOT_COLLISION*quadCollisions + \
               W_GOAL_POS * (ploss).sum() + \
               W_GOAL_VEL * (vloss).sum()

        if VERBOSE and i%100==0:
            print("Iteration %d, Violations: Collision=%d, Final Pos=%d, Final Vel=%d" % (
                i,
                int(quadCollisions),
                int(finalPosInfeasible.sum()),
                int(finalVelInfeasible.sum())
            ), flush=True)

        i+=1
        
        if torch.all(finalPosInfeasible==0) and torch.all(finalVelInfeasible==0) and quadCollisions==0:
            solved=True
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
        p = TimedSolve(start='FACE', goal='FACE')
        if PLOT:
            fig = plt.figure(figsize=(10,10))
            ax = plt.subplot(111)
            ax.set_xlim(np.floor(min(np.min(p[:,:,0]), np.min(p[:,:,1]))/5-0.01)*5, np.ceil(max(np.max(p[:,:,0]), np.max(p[:,:,1]))/5+0.01)*5)
            ax.set_ylim(np.floor(min(np.min(p[:,:,0]), np.min(p[:,:,1]))/5-0.01)*5, np.ceil(max(np.max(p[:,:,0]), np.max(p[:,:,1]))/5+0.01)*5)
            
            NT = p.shape[0]
            NQ = p.shape[1]

            for i in np.arange(NQ)[::-1]:
                plt.plot(p[:,i,0], p[:,i,1], color='k' if i<50 else 'r')
            plt.plot(p[0,:,0], p[0,:,1], 'ko')
            plt.show()