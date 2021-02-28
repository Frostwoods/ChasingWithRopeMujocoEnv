import os
import sys

import time

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

from functionTools.loadSaveModel import saveToPickle,GetSavePath,LoadTrajectories,loadFromPickle,readParametersFromDf
# from functionTools.trajectory import ComputeStatistics#,SampleTrajectory
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



def simuliateOneParameter(parameterDict,evalNum,randomSeed,dt):
    mujocoVisualize = False
    demoVisualize = False


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
    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','variousCollision_numBlocks=1_numSheeps=1_numWolves=1dt={}.xml'.format(dt))

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
    numSimulationFrames = int(0.1/dt)
    print(numSimulationFrames)
    # dt=0.01
    damping=2.5
    reshapeAction=lambda action:action
    # transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt/numSimulationFrames,agentsMaxSpeedList,mujocoVisualize,reshapeAction)


    maxRunningSteps = 10
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
    startTime=time.time()
    for i in range(evalNum):

        np.random.seed(randomSeed+i)
        initSpeedDirection=np.random.uniform(-np.pi/2,np.pi/2,1)[0]
        initSpeed=np.random.uniform(0,1,1)[0]
        initActionDirection=np.random.uniform(-np.pi/2,np.pi/2,1)[0]
        initForce=np.random.uniform(0,5,1)[0]
        # print(initSpeedDirection,initSpeed,initActionDirection,initForce)

        fixReset=ResetFixWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents,numBlocks)
        blocksState=[[0,0,0,0]]#posX posY velX velY
        reset=lambda :fixReset([-0.28,0,8,8],[initSpeed*np.cos(initSpeedDirection),initSpeed*np.sin(initSpeedDirection),0,0],blocksState)

        transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt*numSimulationFrames,agentsMaxSpeedList,mujocoVisualize,isTerminal,reshapeAction)

        initAction=[[initForce*np.cos(initActionDirection),initForce*np.sin(initActionDirection)],[0,0]]
        impulsePolicy=ImpulsePolicy(initAction)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)

        traj = sampleTrajectory(impulsePolicy)
        # print('traj',traj[0])
        trajList.append( traj)

    # saveTraj
    # print(trajList)
    saveTraj = True
    if saveTraj:
        # trajSavePath = os.path.join(dirName,'traj', 'evaluateCollision', 'CollisionMoveTransitDamplingCylinder.pickle')
        trajectorySaveExtension = '.pickle'
        fixedParameters = {'isMujoco': 1,'isCylinder':1,'randomSeed':randomSeed,'evalNum':evalNum}
        trajectoriesSaveDirectory=trajSavePath = os.path.join(dirName,'..', 'trajectory','variousCollsiondt={}'.format(dt))
        generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        trajectorySavePath = generateTrajectorySavePath(parameterDict)
        saveToPickle(trajList, trajectorySavePath)

    # visualize

    if demoVisualize:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        trajToRender = np.concatenate(trajList)
        render(trajToRender)
    endTime = time.time()
    print("Time taken {} seconds to generate {} trajectories".format((endTime - startTime),evalNum))

def drawPerformanceLine(dataDf, axForDraw,lineParameter,prametersToDrop):

    for key, grp in dataDf.groupby(lineParameter):
        grp.index = grp.index.droplevel(prametersToDrop)
        print(grp['mean'].values)
        # grp['agentMean'] = np.array([value for value in grp['mean'].values])
        # grp['agentMean'] =  grp['mean'].values
        # grp['agentStd'] = np.array([value for value in grp['std'].values])
        # grp['agentStd'] = grp['std'].values
        # plt.scatter(range(len(grp['mean'].values[0])),grp['mean'].values[0],s=5,marker='x',label=lineParameter[0]+'={}'.format(key))
        plt.plot(range(len(grp['mean'].values[0])),grp['mean'].values[0],alpha=0.8,label=lineParameter[0]+'={}'.format(key))
        plt.xlabel('timeStep')

        # grp.plot(ax=axForDraw,x=range(5), y='mean', marker='o', label=lineParameter[0]+'={}'.format(key))
        # grp.scatter(ax=axForDraw, y='mean', marker='o', label=lineParameter[0]+'={}'.format(key))
    print(dataDf)
    # dataDf.plot(ax=axForDraw,x=lineParameter, y='mean', marker='o')
class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = np.array(self.measurementFunction(allTrajectories))

        measurementMean = np.mean(allMeasurements,axis=0)
        measurementStd = np.std(allMeasurements,axis=0)

        return pd.Series({'mean': measurementMean, 'std': measurementStd})




def main():
    manipulatedVariables = OrderedDict()
    #solimp
    manipulatedVariables['dmin'] = [0.9]
    manipulatedVariables['dmax'] = [0.9999]
    manipulatedVariables['width'] = [0.001]
    manipulatedVariables['midpoint'] = [0.5]#useless
    manipulatedVariables['power'] = [1]#useless
    #solref
    manipulatedVariables['timeconst'] =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    manipulatedVariables['dampratio'] =[0.2,0.4,0.6,0.8,1.0]

    eavluateParmetersList=['dampratio','dmax']
    lineParameter=['timeconst']
    prametersToDrop=list(set(manipulatedVariables.keys())-set(eavluateParmetersList+lineParameter))
    print(prametersToDrop,manipulatedVariables)

   # timeconst： 0.1-0.2
   # dmin： 0.5-dimax
   # dmax：0.9-0.9999
   # dampratio：0.3-0.8
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    evalNum=500
    randomSeed=1542
    dt=0.05#0.05

    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..')
    trajectoryDirectory = os.path.join(dataFolderName, 'trajectory','variousCollsiondt={}'.format(dt))
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    for parameterOneCondiiton in parametersAllCondtion:
        # print(parameterOneCondiiton,evalNum,randomSeed)
        simuliateOneParameter(parameterOneCondiiton,evalNum,randomSeed,dt)


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    # save evaluation trajectories



    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'isMujoco': 1,'isCylinder':1,'evalNum':evalNum,'randomSeed':randomSeed}


    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories


    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))


    def compareXPos(traj,baselineTraj):
        SE=[np.linalg.norm(s1[0][0][:2]-s2[0][0][:2])**2  for s1,s2 in zip (baselineTraj,traj)]

        return SE

    originEnvTrajSavePath =os.path.join(dirName,'..','trajectory','variousCollsion','collisionMoveOriginBaseLine_evalNum={}_randomSeed={}.pickle'.format(evalNum,randomSeed))
    baselineTrajs=loadFromPickle(originEnvTrajSavePath)

    class CompareTrajectories():
        def __init__(self, baselineTrajs,calculateStatiscs):
            self.baselineTrajs = baselineTrajs
            self.calculateStatiscs = calculateStatiscs
        def __call__(self,trajectorys):


            allMeasurements=np.array([self.calculateStatiscs(traj,baselineTraj) for traj,baselineTraj in zip (trajectorys,self.baselineTrajs)])
            # print()
            measurementMean = allMeasurements.mean(axis=0)
            measurementStd =allMeasurements.std(axis=0)
            return pd.Series({'mean': measurementMean, 'std': measurementStd})

    measurementFunction = CompareTrajectories(baselineTrajs,compareXPos)


    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)



    fig = plt.figure(0)

    rowName,columnName=eavluateParmetersList
    numRows = len(manipulatedVariables[rowName])
    numColumns = len(manipulatedVariables[columnName])
    plotCounter = 1
    selfId=0
    for rowVar, grp in statisticsDf.groupby(rowName):
        grp.index = grp.index.droplevel(rowName)

        for columVar, group in grp.groupby(columnName):
            group.index = group.index.droplevel(columnName)

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns==1:
                axForDraw.set_ylabel(rowName+': {}'.format(rowVar))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(columVar))

            axForDraw.set_ylim(0, 0.02)
            drawPerformanceLine(group, axForDraw,lineParameter,prametersToDrop)
            plotCounter += 1


    plt.suptitle('MSEforPos,dt={}'.format(dt))
    plt.legend(loc='best')
    # plt.show()

    def compareV(traj,baselineTraj):
        # [print(s1[0][0]-s2[0][0],np.linalg.norm(s1[0][0][2:]-s2[0][0][2:]),np.linalg.norm(s1[0][0][:2]-s2[0][0][:2])) for s1,s2 in zip (baselineTraj,traj) ]
        SE=[np.linalg.norm(s1[0][0][2:]-s2[0][0][2:])**2 for s1,s2 in zip (baselineTraj,traj)]
        return SE


    measurementFunction2 = CompareTrajectories(baselineTrajs,compareV)


    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf2)

    fig = plt.figure(1)

    numRows = len(manipulatedVariables[rowName])
    numColumns = len(manipulatedVariables[columnName])
    plotCounter = 1
    for rowVar, grp in statisticsDf2.groupby(rowName):
        grp.index = grp.index.droplevel(rowName)

        for columVar, group in grp.groupby(columnName):
            group.index = group.index.droplevel(columnName)

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns==1:
                axForDraw.set_ylabel(rowName+': {}'.format(rowVar))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(columVar))

            axForDraw.set_ylim(0, 0.4)
            drawPerformanceLine(group, axForDraw,lineParameter,prametersToDrop)
            plotCounter += 1


    plt.suptitle('MSEforSpeed,dt={}'.format(dt))
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
    # simuliateOneParameter()
