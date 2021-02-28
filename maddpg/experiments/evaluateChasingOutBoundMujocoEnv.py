import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from collections import OrderedDict
import itertools as it
import json
import xmltodict
import mujoco_py as mujoco
import pandas as pd

from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.chasingEnv.multiAgentEnv import ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from maddpg.maddpgAlgor.trainer.myMADDPG import *

from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos,SampleBlockState,IsOverlap
from functionTools.loadSaveModel import saveToPickle,GetSavePath,LoadTrajectories,loadFromPickle,readParametersFromDf

wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000


def simuliateOneParameter(parameterOneCondiiton,evalNum,randomSeed):
    saveTraj = True
    visualizeTraj = False
    visualizeMujoco = False

    numWolves = parameterOneCondiiton['numWolves']
    numSheeps = parameterOneCondiiton['numSheeps']
    numBlocks = parameterOneCondiiton['numBlocks']
    timeconst= parameterOneCondiiton['timeconst']
    dampratio= parameterOneCondiiton['dampratio']
    maxTimeStep = parameterOneCondiiton['maxTimeStep']
    sheepSpeedMultiplier = parameterOneCondiiton['sheepSpeedMultiplier']
    individualRewardWolf = parameterOneCondiiton['individualRewardWolf']
    hasWalls = parameterOneCondiiton['hasWalls']
    dt = parameterOneCondiiton['dt']



    print("maddpg: {} wolves, {} sheep, {} blocks, saveTraj: {}, visualize: {}".format(numWolves, numSheeps, numBlocks, str(saveTraj), str(visualizeTraj)))


    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks

    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)

    if individualRewardWolf:
        rewardWolf = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    else:
        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)

    rewardFunc = lambda state, action, nextState: list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))
    dirName = os.path.dirname(__file__)

    if hasWalls:

        physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','dt={}'.format(dt),'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.format(hasWalls,numBlocks,numSheeps,numWolves,timeconst,dampratio))


    else:
        physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.
          format(numBlocks,numSheeps,numWolves,timeconst,dampratio))

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    envXml=xmltodict.unparse(envXmlDict)

    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)
    # print(physicsSimulation.model.body_pos)

    qPosInit = [0, 0]*numAgents
    qVelInit = [0, 0]*numAgents
    qVelInitNoise = 0*hasWalls
    qPosInitNoise = 0.8*hasWalls
    getBlockRandomPos = lambda: np.random.uniform(-0.7*hasWalls, +0.7*hasWalls, 2)
    getBlockSpeed = lambda: np.zeros(2)

    numQPos=len(physicsSimulation.data.qpos)
    numQVel = len(physicsSimulation.data.qvel)

    sampleAgentsQPos=lambda: np.asarray(qPosInit)+np.random.uniform(low=-qPosInitNoise, high=qPosInitNoise, size=numQPos)
    sampleAgentsQVel=lambda: np.asarray(qVelInit) + np.random.uniform(low=-qVelInitNoise, high=qVelInitNoise, size=numQVel)


    minDistance=0.2+2*blockSize
    isOverlap=IsOverlap(minDistance)

    sampleBlockState=SampleBlockState(numBlocks,getBlockRandomPos,getBlockSpeed,isOverlap)

    reset=ResetUniformWithoutXPos(physicsSimulation,  numAgents, numBlocks,sampleAgentsQPos, sampleAgentsQVel,sampleBlockState)

    damping=2.5
    numSimulationFrames =int(0.1/dt)
    agentsMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps
    reshapeAction = ReshapeAction()
    isTerminal = lambda state: False
    transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt*numSimulationFrames,agentsMaxSpeedList,visualizeMujoco,isTerminal,reshapeAction)


    maxRunningStepsToSample = 100
    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]



    if hasWalls:



        modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPGeavluateWall','dt={}'.format(dt),'hasWalls='+str(hasWalls)+'numBlocks='+str(numBlocks)+'numSheeps='+str(numSheeps)+'numWolves='+str(numWolves)+'timeconst='+str(timeconst)+'dampratio='+str(dampratio)+'individualRewardWolf='+str(individualRewardWolf)+'sheepSpeedMultiplier='+str(sheepSpeedMultiplier))
        individStr = 'individ' if individualRewardWolf else 'shared'
        fileName = "maddpghasWalls={}{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(hasWalls,numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)


    else :
        modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG','numBlocks='+str(numBlocks)+'numSheeps='+str(numSheeps)+'numWolves='+str(numWolves)+'timeconst='+str(timeconst)+'dampratio='+str(dampratio))
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0shared_agent".format(numWolves, numSheeps, numBlocks)
    modelPaths = [os.path.join(modelFolder,  fileName + str(i) + '60000eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    startTime=time.time()
    trajList = []
    for i in range(evalNum):
        np.random.seed(i)
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))
    endTime = time.time()
    print("Time taken {} seconds to generate {} trajectories".format((endTime - startTime),evalNum))


    # saveTraj
    if saveTraj:

        trajectoryFolder=os.path.join(dirName, '..', 'trajectory', 'evluateWall')
        if not os.path.exists(trajectoryFolder):
            os.makedirs(trajectoryFolder)

        trajectorySaveExtension = '.pickle'
        fixedParameters = {'randomSeed':randomSeed,'evalNum':evalNum}
        generateTrajectorySavePath = GetSavePath(trajectoryFolder, trajectorySaveExtension, fixedParameters)
        trajectorySavePath = generateTrajectorySavePath(parameterOneCondiiton)


        saveToPickle(trajList, trajectorySavePath)


    # visualize
    if visualizeTraj:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        print(trajList[0][0],'!!!3',trajList[0][1])
        trajToRender = np.concatenate(trajList)
        render(trajToRender)


    return endTime - startTime
def main():


    manipulatedVariables = OrderedDict()

    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheeps'] = [1]
    manipulatedVariables['numBlocks'] = [2]
    manipulatedVariables['maxTimeStep'] = [25]
    manipulatedVariables['sheepSpeedMultiplier'] = [1.0]
    manipulatedVariables['individualRewardWolf'] = [0]
    manipulatedVariables['timeconst'] = [0.5]
    manipulatedVariables['dampratio'] = [0.2]
    manipulatedVariables['hasWalls'] = [1.0,1.5,2.0]
    manipulatedVariables['dt'] = [0.05]



    eavluateParmetersList=['hasWalls','dt']
    lineParameter=['sheepSpeedMultiplier']
    prametersToDrop=list(set(manipulatedVariables.keys())-set(eavluateParmetersList+lineParameter))


    print(prametersToDrop,manipulatedVariables)



    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    evalNum=500
    randomSeed=1542
    dt=0.05#0.05

    dirName = os.path.dirname(__file__)

    trajectoryDirectory=os.path.join(dirName, '..', 'trajectory', 'evluateWall')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    time=[]
    for parameterOneCondiiton in parametersAllCondtion:
        # print(parameterOneCondiiton,evalNum,randomSeed)

        sampletime=simuliateOneParameter(parameterOneCondiiton,evalNum,randomSeed)
        time.append(sampletime)



    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    # save evaluation trajectories



    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = {'randomSeed':randomSeed,'evalNum':evalNum}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories


    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))

    class CheckStateOfofBound(object):
        """docstring for ClassName"""
        def __init__(self, minDistance):
            self.minDistance=minDistance
        def __call__(self,state):
            absloc=np.abs([agent[:2] for agent in   state[0][:-2]])
            return np.any(absloc> self.minDistance)

    class ComputeOutOfBounds:
        def __init__(self, getTrajectories, CheckStateOfofBound):
            self.getTrajectories = getTrajectories
            self.CheckStateOfofBound = CheckStateOfofBound

        def __call__(self, oneConditionDf):
            allTrajectories = self.getTrajectories(oneConditionDf)
            minDistance=readParametersFromDf(oneConditionDf)['hasWalls']+0.05

            checkStateOfofBound=CheckStateOfofBound(minDistance)
            # allMeasurements = np.array(self.measurementFunction(allTrajectories,minDistance))
            # allMeasurements = np.array(self.measurementFunction(allTrajectories,minDistance))s
            allMeasurements=np.array([checkStateOfofBound(state) for traj in allTrajectories for state in traj ])
            # print(allMeasurements)
            measurementMean = np.mean(allMeasurements)
            measurementStd = np.std(allMeasurements)
            return pd.Series({'mean': measurementMean, 'std': measurementStd})

    computeStatistics = ComputeOutOfBounds(loadTrajectoriesFromDf, CheckStateOfofBound)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)

    for i,parameterOneCondiiton in enumerate(parametersAllCondtion):
        print(parameterOneCondiiton,time[i])
if __name__ == '__main__':
    main()
