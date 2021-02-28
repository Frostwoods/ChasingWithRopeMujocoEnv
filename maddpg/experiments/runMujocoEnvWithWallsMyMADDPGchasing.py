import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json
import xmltodict
import mujoco_py as mujoco

from maddpg.maddpgAlgor.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import  ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos,SampleBlockState,IsOverlap

# fixed training parameters
maxEpisode = 60000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numBlocks saveAllmodels = True or False

def main():
    debug = 0
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        timeconst=0.5 #solref[0] of each body  in mujoco xml
        dampratio=0.2 #solref[1] of each body  in mujoco xml
        saveAllmodels = True
        maxTimeStep = 25
        sheepSpeedMultiplier = 1.0
        individualRewardWolf = int(False)
        hasWalls=2.0
        dt=0.02
    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        timeconst = float(condition['timeconst'])
        dampratio = float(condition['dampratio'])
        dt =float(condition['dt'])
        maxTimeStep = int(condition['maxTimeStep'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = int(condition['individualRewardWolf'])
        hasWalls=float(condition['hasWalls'])
        saveAllmodels = True
    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps, sheepSpeed: {}x, wolfIndividualReward: {}, save all models: {}".
          format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individualRewardWolf, str(saveAllmodels)))
    print('timeconst',timeconst,'dampratio',dampratio)

    modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPGeavluateWall','dt={}'.format(dt),'hasWalls='+str(hasWalls)+'numBlocks='+str(numBlocks)+'numSheeps='+str(numSheeps)+'numWolves='+str(numWolves)+'timeconst='+str(timeconst)+'dampratio='+str(dampratio)+'individualRewardWolf='+str(individualRewardWolf)+'sheepSpeedMultiplier='+str(sheepSpeedMultiplier))

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)


    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numWolves + numSheeps))
    mastersID = list(range(numWolves + numSheeps, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    agentsMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps

    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound =lambda state :0 #PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound)
    if individualRewardWolf:
        rewardWolf = RewardWolfIndividual(wolvesID, sheepsID, entitiesSizeList, isCollision)
    else:
        rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    # if haswall:
    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','dt={}'.format(dt),'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.format(hasWalls,numBlocks,numSheeps,numWolves,timeconst,dampratio))
    # else:
    #     physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.
    #       format(numBlocks,numSheeps,numWolves,timeconst,dampratio))
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

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


    minDistance=0.2+2*blockSize#>2*wolfSize+2*blockSize
    isOverlap=IsOverlap(minDistance)

    sampleBlockState=SampleBlockState(numBlocks,getBlockRandomPos,getBlockSpeed,isOverlap)

    reset=ResetUniformWithoutXPos(physicsSimulation,  numAgents, numBlocks,sampleAgentsQPos, sampleAgentsQVel,sampleBlockState)



    damping=2.5
    numSimulationFrames = int(0.1/dt)
    visualize=False
    isTerminal = lambda state: [False]* numAgents
    reshapeAction = ReshapeAction()
    transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt*numSimulationFrames,agentsMaxSpeedList,visualize,isTerminal,reshapeAction)


    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, mastersID, getPosFromAgentState,getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsList)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsList]

    sampleOneStep = SampleOneStep(transit, rewardFunc)
    runDDPGTimeStep = RunTimeStep(actOneStep, sampleOneStep, trainMADDPGModels, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in range(numAgents)]
    modelSaveRate = 1000
    individStr = 'individ' if individualRewardWolf else 'shared'
    fileName = "maddpghasWalls={}{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(hasWalls,numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)

    modelPath = os.path.join(modelFolder, fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = maddpg(replayBuffer)


if __name__ == '__main__':
    main()


