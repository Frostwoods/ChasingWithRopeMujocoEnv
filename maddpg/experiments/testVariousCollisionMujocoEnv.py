import os
import sys



os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

import xmltodict
import numpy as np
import itertools as it
import mujoco_py as mujoco
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt

from functionTools.loadSaveModel import saveToPickle,GetSavePath,LoadTrajectories,loadFromPickle
# from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *


from environment.chasingEnv.multiAgentEnv import RewardWolf,RewardSheep,IsCollision,getPosFromAgentState,getVelFromAgentState,PunishForOutOfBound,Observe
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos,ResetFixWithoutXPos

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
def transferNumberListToStr(numList):
    strList=[str(num) for num in numList]
    return ' '.join(strList)


class MakePropertyList():
    def __init__(self, transferNumberListToStr):
        self.transferNumberListToStr=transferNumberListToStr
    def __call__(self,idlist,keyNameList,valueList):
        propertyDict={}
        [propertyDict.update({geomid:{name:self.transferNumberListToStr(value) for name, value  in zip (keyNameList,values)}}) for geomid,values in zip(idlist,valueList) ]
        return propertyDict


def changeGeomProperty(envDict,geomPropertyDict):
    for number,propertyDict in geomPropertyDict.items():
        for name,value in propertyDict.items():
            envDict['mujoco']['worldbody']['body'][number]['geom'][name]=value

    return envDict


def simuliateOneParameter(parameterDict):


    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2


    wolfColor = np.array([0.85, 0.35, 0.35])
    sheepColor = np.array([0.35, 0.85, 0.35])
    blockColor = np.array([0.25, 0.25, 0.25])

    wolvesID = [0]
    sheepsID = [1]
    blocksID = [2]

    numWolves = len(wolvesID)
    numSheeps = len(sheepsID)
    numBlocks = len(blocksID)

    sheepMaxSpeed = 1.3
    wolfMaxSpeed =1.0
    blockMaxSpeed = None

    mujocoVisualize=False

    agentsMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps

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

    makePropertyList=MakePropertyList(transferNumberListToStr)

    #changeCollisionReleventParameter
    dmin=parameterDict['dmin']
    dmax=parameterDict['dmax']
    width=parameterDict['width']
    midpoint=parameterDict['midpoint']
    power=parameterDict['power']
    timeconst=parameterDict['timeconst']
    dampratio=parameterDict['dampratio']

    geomIds=[1,2,3]
    keyNameList=['@solimp','@solref']
    valueList=[[[dmin,dmax,width,midpoint,power],[timeconst,dampratio]]]*len(geomIds)
    collisionParameter=makePropertyList(geomIds,keyNameList,valueList)

#changeSize
    # geomIds=[1,2]
    # keyNameList=['@size']
    # valueList=[[[0.075,0.075]],[[0.1,0.1]]]
    # sizeParameter=makePropertyList(geomIds,keyNameList,valueList)

#load env xml and change some geoms' property
    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','variousCollision_numBlocks=1_numSheeps=0_numWolves=1.xml')

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())

    envXmlPropertyDictList=[collisionParameter]
    changeEnvXmlPropertFuntionyList=[changeGeomProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

    envXml=xmltodict.unparse(envXmlDict)
    # print(envXmlDict['mujoco']['worldbody']['body'][0])
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)


    # MDP function
    qPosInit = [0, 0]*numAgents
    qVelInit = [0, 0]*numAgents

    qVelInitNoise = 0
    qPosInitNoise = 1

    # reset=ResetUniformWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents, numBlocks,qPosInitNoise, qVelInitNoise)

    # fixReset=ResetFixWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents,numBlocks)
    # blocksState=[[0,-0.8,0,0]]
    # reset=lambda :fixReset([-0.5,0.8,-0.5,0,0.5,0.8],[0,0,0,0,0,0],blocksState)


    isTerminal = lambda state: False
    numSimulationFrames = 1

    # physicsViewer=None
    dt=0.1
    damping=2.5
    reshapeAction=lambda action:action
    # transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt/numSimulationFrames,agentsMaxSpeedList,mujocoVisualize,reshapeAction)


    maxRunningSteps = 5
    # sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)
    # observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    # observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    # initObsForParams = observe(reset())
    # obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    class ImpulsePolicy(object):
        """docstring for c"""
        def __init__(self,initAction):
            self.initAction = initAction
        def __call__(self,state,timeStep):
            action=[[0,0],[0,0]]
            if timeStep==0:
                action=self.initAction
            return action

    worldDim = 2
    trajList = []
    evalNum=10
    randomSeed=9527
    for i in range(evalNum):

        np.random.seed(randomSeed+i)

        # policy =lambda state: [[-3,0] for agent in range(numAgents)]
        # np.random.seed(i)
        # policy =lambda state: [np.random.uniform(-5,5,2) for agent in range(numAgents)]
        # policy =lambda state: [[0,5] for agent in range(numAgents)]
        # policy =lambda state: [[1,0],[1,0],[-1,0]]
        # policy =lambda state: [[np.random.uniform(0,1,1),0] ,[np.random.uniform(-1,0,1),0] ]
        initSpeedDirection=np.random.uniform(-np.pi/2,np.pi/2,1)
        initSpeed=np.random.uniform(0,1,1)[0]
        # reset=lambda :np.array([[-0.28,0,initSpeed*np.cos(initSpeedDirection)[0],initSpeed*np.sin(initSpeedDirection)[0]],[8,8,0,0],[0,0,0,0]])
        fixReset=ResetFixWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents,numBlocks)
        blocksState=[[0,0,0,0]]#posX posY velX velY
        reset=lambda :fixReset([-0.28,0,8,8],[initSpeed*np.cos(initSpeedDirection)[0],initSpeed*np.sin(initSpeedDirection)[0],0,0],blocksState)
        isTerminal=lambda state:False
        transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt*numSimulationFrames,agentsMaxSpeedList,mujocoVisualize,isTerminal,reshapeAction)

        initActionDirection=np.random.uniform(-np.pi/2,np.pi/2,1)
        initForce=np.random.uniform(0,5,1)[0]
        initAction=[[initForce*np.cos(initActionDirection)[0],initForce*np.sin(initActionDirection)[0]],[0,0]]
        print(initAction,reset())
        impulsePolicy=ImpulsePolicy(initAction)


        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

        traj = sampleTrajectory(impulsePolicy)
        # print('traj',[tr[0][1][0]-tr[0][0][0] for tr in traj])
        trajList = trajList + list(traj)

    # saveTraj
        saveTraj = True
        if saveTraj:
            # trajSavePath = os.path.join(dirName,'traj', 'evaluateCollision', 'CollisionMoveTransitDamplingCylinder.pickle')
            trajectorySaveExtension = '.pickle'
            fixedParameters = {'isMujoco': 1,'isCylinder':1}
            trajectoriesSaveDirectory=trajSavePath = os.path.join(dirName,'..', 'trajectory','variousCollsion')
            generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            trajectorySavePath = generateTrajectorySavePath(parameterDict)
            saveToPickle(traj, trajectorySavePath)

    # visualize

    # physicsViewer.render()
    visualize = False
    if visualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajList)

def main():
    manipulatedVariables = OrderedDict()
    #solimp
    manipulatedVariables['dmin'] = [0.9]
    manipulatedVariables['dmax'] = [0.9999]
    manipulatedVariables['width'] = [0.001]
    manipulatedVariables['midpoint'] = [0.5]#useless
    manipulatedVariables['power'] = [1]#useless
    #solref
    manipulatedVariables['timeconst'] = [0.02,0.4]#[0.1,0.2,0.4,0.6]
    manipulatedVariables['dampratio'] = [0.5]


   # timeconst： 0.1-0.2
   # dmin： 0.5-dimax
   # dmax：0.9-0.9999
   # dampratio：0.3-0.8


    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    for parameterOneCondiiton in parametersAllCondtion:
        print(parameterOneCondiiton)
        simuliateOneParameter(parameterOneCondiiton)


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..')
    trajectoryDirectory = os.path.join(dataFolderName, 'trajectory','variousCollsion')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)


    # originEnvTrajSavePath =os.path.join(dirName,'traj','evaluateCollision','collisionMoveOriginBaseLine.pickle')
    originEnvTrajSavePath =os.path.join(dirName,'..','trajectory','variousCollsion','collisionMoveOriginBaseLine_evalNum=11_randomSeed=542.pickle')
    originTraj=loadFromPickle(originEnvTrajSavePath)[0]
    originTrajdata=[[],[],[],[]]

    originTrajdata[0]=[s[0][0][:2] for s in originTraj]
    originTrajdata[1]=[s[0][1][:2] for s in originTraj]
    originTrajdata[2]=[s[0][0][2:] for s in originTraj]
    originTrajdata[3]=[s[0][1][2:] for s in originTraj]
    originTraj=loadFromPickle(originEnvTrajSavePath)

    trajectoryExtension = '.pickle'
    fixedParameters = {'isMujoco': 1,'isCylinder':1}

    generateTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, fixedParameters)

    getparameter= GetSavePath('', '', {})
    def parseTraj(traj):
        x=[s[0] for s in traj]
        y=[s[1] for s in traj]
        return x,y
    for i, parameterOneCondiiton in enumerate(parametersAllCondtion):
        mujocoTrajdata=[[],[],[],[]]

        trajectorySavePath = generateTrajectorySavePath(parameterOneCondiiton)
        mujocoTraj=loadFromPickle(trajectorySavePath)
        mujocoTrajdata[0]=[s[0][0][:2] for s in mujocoTraj]
        mujocoTrajdata[1]=[s[0][1][:2] for s in mujocoTraj]
        mujocoTrajdata[2]=[s[0][0][2:] for s in mujocoTraj]
        mujocoTrajdata[3]=[s[0][1][2:] for s in mujocoTraj]
        fig = plt.figure(i)
        numRows = 2
        numColumns = 2
        plotCounterNum = numRows*numColumns
        subName=['wolf0Pos(vs.Sheep)','wolf1Pos(vs.None)','wolf0Vel(vs.Sheep)','wolf1Vel(vs.None)']
        for plotCounter in range(plotCounterNum):

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter+1)


            axForDraw.set_title(subName[plotCounter])
            # axForDraw.set_ylim(-0.5, 1)
            # axForDraw.set_xlim(-1, 1)
            originX,originY=parseTraj(originTrajdata[plotCounter])
            mujocoX,mujocoY=parseTraj(mujocoTrajdata[plotCounter])

            plt.scatter(range(len(originX)),originX,s=5,c='red',marker='x',label='origin')
            plt.scatter(range(len(mujocoX)),mujocoX,s=5,c='blue',marker='v',label='mujoco')

        numSimulationFrames=1
        plt.suptitle(getparameter(parameterOneCondiiton)+'numSimulationFrames={}'.format(numSimulationFrames))
        # plt.suptitle('OriginEnv')
        plt.legend(loc='best')

    plt.show()

if __name__ == '__main__':
    main()
    # simuliateOneParameter()
