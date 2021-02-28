import os
import sys
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from environment.chasingEnv.multiAgentEnv import *
from functionTools.loadSaveModel import saveToPickle
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *

from gym import spaces



def main():
    # wolvesID = [0, 1]
    # sheepsID = [2]
    # blocksID = [3]
    visualize = False
    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2

    sheepMaxSpeed = 1.3
    wolfMaxSpeed = 1.0
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
    reset=lambda :np.array([[-0.5,0,0,0],[8,8,0,0],[0,-8,0,0]])
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
    maxRunningSteps = 100
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2

    trajList = []
    evalNum=1
    for i in range(evalNum):

        # policy =lambda state: [s[-3,0] for agent in range(numAgents)]
        np.random.seed(i)
        # policy =lambda state: [np.random.uniform(-5,5,2) for agent in range(numAgents)]
        # policy =lambda state: [[1,1] for agent in range(numAgents)]
        # policy =lambda state: [[float(np.random.uniform(0,1,1)),0] ,[float(np.random.uniform(-1,0,1)),0] ]
        policy =lambda state: [[1,0],[0,0] ]
        traj = sampleTrajectory(policy)
        # print(i,'traj',[state[1] for state in traj)
        # print('traj',[tr[0][0][0] for tr in traj])
        print('traj',traj[0])
        trajList.append( traj)

    # saveTraj
    saveTraj = True
    # print(trajList[0][0][0])
    if saveTraj:
        trajSavePath = os.path.join(dirName,'..','trajectory',  'LineMoveOriginBaseLine.pickle')
        saveToPickle(trajList, trajSavePath)
    # print('traj',trajList[25])
    # visualize

    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)


if __name__ == '__main__':
    main()
