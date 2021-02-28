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

from functionTools.loadSaveModel import saveToPickle,GetSavePath,LoadTrajectories,loadFromPickle,readParametersFromDf
from functionTools.trajectory import SampleTrajectory,ComputeStatistics
from visualize.visualizeMultiAgent import *


from environment.chasingEnv.multiAgentEnv import RewardWolf,RewardSheep,IsCollision,getPosFromAgentState,getVelFromAgentState,PunishForOutOfBound,Observe
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos,ResetFixWithoutXPos

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

    wolvesID = [0,1]
    sheepsID = [2]
    blocksID = [3]

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

    geomIds=[1,2]
    keyNameList=['@solimp','@solref']
    valueList=[[[dmin,dmax,width,midpoint,power],[timeconst,dampratio]]]*len(geomIds)
    collisionParameter=makePropertyList(geomIds,keyNameList,valueList)

#changeSize
    # geomIds=[1,2]
    # keyNameList=['@size']
    # valueList=[[[0.075,0.075]],[[0.1,0.1]]]
    # sizeParameter=makePropertyList(geomIds,keyNameList,valueList)

#load env xml and change some geoms' property
    # physicsDynamicsPath=os.path.join(dirName,'multiAgentcollision.xml')
    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','multiAgentcollision.xml')
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    print(envXmlDict)
    envXmlPropertyDictList=[collisionParameter]
    changeEnvXmlPropertFuntionyList=[changeGeomProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

    envXml=xmltodict.unparse(envXmlDict)

    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)


    # MDP function
    qPosInit = [0, 0]*numAgents
    qVelInit = [0, 0]*numAgents

    qVelInitNoise = 0
    qPosInitNoise = 1

    # reset=ResetUniformWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents, numBlocks,qPosInitNoise, qVelInitNoise)

    fixReset=ResetFixWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents,numBlocks)
    blocksState=[[0,0,0,0]]
    reset=lambda :fixReset([-0.5,0.8,-0.5,0,0.5,0.8],[0,0,0,0,0,0],blocksState)


    isTerminal = lambda state: False
    numSimulationFrames = 1
    visualize=False
    # physicsViewer=None
    dt=0.1
    damping=2.5
    transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt/numSimulationFrames,agentsMaxSpeedList,visualize,isTerminal)


    maxRunningSteps = 100
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transit, isTerminal, rewardFunc, reset)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2
    evalNum=1

    trajectorySaveExtension = '.pickle'
    fixedParameters = {'isMujoco': 1,'isCylinder':1,'evalNum':evalNum}
    trajectoriesSaveDirectory=trajSavePath = os.path.join(dirName,'..','trajectory')
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parameterDict)


    if not os.path.isfile(trajectorySavePath):
        trajList =list()

        for i in range(evalNum):

            # policy =lambda state: [[-3,0] for agent in range(numAgents)]
            np.random.seed(i)
            # policy =lambda state: [np.random.uniform(-5,5,2) for agent in range(numAgents)]sss
            # policy =lambda state: [[0,1] for agent in range(numAgents)]
            policy =lambda state: [[1,0] ]
            # policy =lambda state: [[np.random.uniform(0,1,1),0] ,[np.random.uniform(-1,0,1),0] ]
            traj = sampleTrajectory(policy)
            # print(i,'traj',[state[1] for state in traj[:2]])
            # print(traj)
            trajList.append( traj)

        # saveTraj
        saveTraj = True
        if saveTraj:
            # trajSavePath = os.path.join(dirName,'traj', 'evaluateCollision', 'CollisionMoveTransitDamplingCylinder.pickle')

            saveToPickle(trajList, trajectorySavePath)

        # visualize

        # physicsViewer.render()
        visualize = True
        if visualize:
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
            render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
            render(trajList)


def drawPerformanceLine(dataDf, axForDraw,xName,lineParameter,prametersToDrop):

    for key, grp in dataDf.groupby(lineParameter):
        grp.index = grp.index.droplevel(prametersToDrop+lineParameter)
        # grp['agentMean'] = np.array([value for value in grp['mean'].values])
        grp['agentMean'] =  grp['mean'].values
        grp['agentStd'] = np.array([value for value in grp['std'].values])
        grp['agentStd'] = grp['std'].values
        grp.plot(ax=axForDraw, y='mean', marker='o', label=lineParameter[0]+'={}'.format(key))
    # print(dataDf)
    # dataDf.plot(ax=axForDraw,x=lineParameter, y='mean', marker='o')
def main():
    manipulatedVariables = OrderedDict()
    #solimp
    manipulatedVariables['dmin'] = [0.8]
    manipulatedVariables['dmax'] = [0.9999]
    manipulatedVariables['width'] = [0.001]
    manipulatedVariables['midpoint'] = [0.5]#useless
    manipulatedVariables['power'] = [1]#useless
    #solref
    manipulatedVariables['timeconst'] = [0.15]
    manipulatedVariables['dampratio'] = [0.4]
    ###
    # timeconst： 0.1-0.2
    # dmin： 0.5-dimax
    # dmax：0.9-0.9999
    # dampratio：0.3-0.8

    ###
    eavluateParmetersList=['dmin','dmax','dampratio']
    lineParameter=['timeconst']
    prametersToDrop=list(set(manipulatedVariables.keys())-set(eavluateParmetersList+lineParameter))
    print(prametersToDrop,manipulatedVariables)
    # manipulatedVariables['timeconst'] = [-100]
    # manipulatedVariables['dampratio'] = [-120]
   # #solref
   #  manipulatedVariables['stiffness'] = [150]
   #  manipulatedVariables['damping'] = [150]
    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..')
    trajectoryDirectory = os.path.join(dataFolderName, 'trajectory')
    if not os.path.exists(trajectoryDirectory):
       os.makedirs(trajectoryDirectory)



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


    evalNum=1
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'isMujoco': 1,'isCylinder':1,'evalNum':evalNum}


    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories


    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))


    def compareXPos(traj,baselineTraj):

        SE=[(np.linalg.norm(s1[0][0][:2]-s2[0][0][:2])**2+np.linalg.norm(s1[0][1][:2]-s2[0][1][:2])**2)/2*100000  for s1,s2 in zip (baselineTraj,traj)]
        return SE

    originEnvTrajSavePath =os.path.join(dirName,'..','trajectory','collisionMoveOriginBaseLine.pickle')
    baselineTrajs=loadFromPickle(originEnvTrajSavePath)

    class CompareTrajectories():
        def __init__(self, baselineTrajs,calculateStatiscs):
            self.baselineTrajs = baselineTrajs
            self.calculateStatiscs = calculateStatiscs
        def __call__(self,trajectorys):

            allMeasurements=[self.calculateStatiscs(traj,baselineTraj) for traj,baselineTraj in zip (trajectorys,self.baselineTrajs)]
            return allMeasurements

    measurementFunction = CompareTrajectories(baselineTrajs,compareXPos)


    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)



    fig = plt.figure(0)

    rowName,columnName,xName=eavluateParmetersList
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

            # axForDraw.set_ylim(90, 200)
            drawPerformanceLine(group, axForDraw, xName,lineParameter,prametersToDrop)
            plotCounter += 1


    plt.suptitle('MSEforPos')
    plt.legend(loc='best')
    # plt.show()

    def compareV(traj,baselineTraj):

        SE=[(np.linalg.norm(s1[0][0][2:]-s2[0][0][2:])**2+np.linalg.norm(s1[0][1][2:]-s2[0][1][2:])**2)/2 *100000 for s1,s2 in zip (baselineTraj,traj)]
        return SE


    measurementFunction2 = CompareTrajectories(baselineTrajs,compareV)


    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf2)

    fig = plt.figure(1)

    numRows = len(manipulatedVariables[rowName])
    numColumns = len(manipulatedVariables[columnName])
    plotCounter = 1
    for rowVar, grp in statisticsDf.groupby(rowName):
        grp.index = grp.index.droplevel(rowName)

        for columVar, group in grp.groupby(columnName):
            group.index = group.index.droplevel(columnName)

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns==1:
                axForDraw.set_ylabel(rowName+': {}'.format(rowVar))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(columVar))

            # axForDraw.set_ylim(90, 200)
            drawPerformanceLine(group, axForDraw, xName,lineParameter,prametersToDrop)
            plotCounter += 1


    plt.suptitle('MSEforSpeed')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
    # simuliateOneParameter()
