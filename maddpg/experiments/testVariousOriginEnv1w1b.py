import os
import sys
import argparse
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle
# from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

from gym import spaces

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            action = policy(state,runningStep)
            # print(action)
            nextState = self.transit(state, action)

            reward = self.rewardFunc(state, action, nextState)
            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward

            trajectory.append((state, action, reward, nextState))

            state = nextState
        # print('epsReward: ', epsReward)
        return trajectory

def main():
    # wolvesID = [0, 1]
    # sheepsID = [2]
    # blocksID = [3]

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2

    sheepMaxSpeed = None
    wolfMaxSpeed = None
    blockMaxSpeed = None

    wolfColor = np.array([0.85, 0.35, 0.35])
    sheepColor = np.array([0.35, 0.85, 0.35])
    blockColor = np.array([0.25, 0.25, 0.25])



    wolvesID = [0]
    sheepsID = [1]
    blocksID = [2]

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks

    entitiesSizeList = [wolfSize]* numWolves + [sheepSize] * numSheeps + [blockSize]* numBlocks
    entityMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed]* numBlocks
    entitiesMovableList = [True]* numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)


    # reset=lambda :np.array([[-0.5,0,0,0],[0.5,0,0,0]])

    # reset=lambda :np.array([[-0.5,0,0,0],[0,0,0,0]])
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    # reshapeAction = ReshapeAction()
    reshapeAction = lambda action:action

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: False
    maxRunningSteps = 10
    # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2


    trajList = []
    evalNum=500

    class ImpulsePolicy(object):
        """docstring for c"""
        def __init__(self,initAction):
            self.initAction = initAction
        def __call__(self,state,timeStep):
            action=[[0,0],[0,0]]
            if timeStep==0:
                action=self.initAction
            return action


    randomSeed=3542
    startTime=time.time()
    for i in range(evalNum):

        # policy =lambda state: [[-3,0] for agent in range(numAgents)]
        np.random.seed(randomSeed+i)
        initSpeedDirection=np.random.uniform(-np.pi/2,np.pi/2,1)[0]
        initSpeed=np.random.uniform(0,1,1)[0]
        initActionDirection=np.random.uniform(-np.pi/2,np.pi/2,1)[0]
        initForce=np.random.uniform(0,5,1)[0]
        # print(initSpeedDirection,initSpeed,initActionDirection,initForce)
        reset=lambda :np.array([[-0.28,0,initSpeed*np.cos(initSpeedDirection),initSpeed*np.sin(initSpeedDirection)],[8,8,0,0],[0,0,0,0]])

        initAction=[[initForce*np.cos(initActionDirection),initForce*np.sin(initActionDirection)],[0,0]]
        # print(initAction,reset())
        impulsePolicy=ImpulsePolicy(initAction)



        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

        traj = sampleTrajectory(impulsePolicy)
        # print('traj',traj[0])
        trajList.append( traj)


    # saveTraj
    saveTraj = True
    # print(trajList[0][0][0])
    if saveTraj:
        trajSavePath = os.path.join(dirName,'..','trajectory','variousCollsionWithoutSpeedLimit',  'collisionMoveOriginBaseLine_evalNum={}_randomSeed={}.pickle'.format(evalNum,randomSeed))
        saveToPickle(trajList, trajSavePath)
    # print('traj',trajList[25])
    # visualize
    visualize = False
    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)

    endTime = time.time()
    print("Time taken {} seconds to generate {} trajectories".format((endTime - startTime),evalNum))

if __name__ == '__main__':
    main()
