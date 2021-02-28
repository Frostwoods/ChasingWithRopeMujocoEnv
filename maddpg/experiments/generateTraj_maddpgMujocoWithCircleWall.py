import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from functionTools.loadSaveModel import saveToPickle, restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.chasingEnv.multiAgentEnv import ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json
import xmltodict
import mujoco_py as mujoco
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos


wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000


def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheeps = 1
        numBlocks = 2
        timeconst=0.5
        dampratio=0.2
        saveTraj = False
        visualizeTraj = True
        maxTimeStep = 100
        sheepSpeedMultiplier = 1.0
        individualRewardWolf = 0
        hasWalls=1
        visualizeMujoco=True

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheeps = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])
        saveTraj = True
        visualizeTraj = False
        maxTimeStep = 50
        sheepSpeedMultiplier = 1.25
        individualRewardWolf = 1

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
    # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','numBlocks=1_numSheeps=1_numWolves=1.xml')
    if hasWalls:
        # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','hasWalls=1_numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.format(numBlocks,numSheeps,numWolves,timeconst,dampratio))

        physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','CicleWallTest3w1s2b.xml')


    else:
        physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','numBlocks={}_numSheeps={}_numWolves={}timeconst={}dampratio={}.xml'.
          format(numBlocks,numSheeps,numWolves,timeconst,dampratio))

    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())
    envXml=xmltodict.unparse(envXmlDict)

    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)
    print(physicsSimulation.model.body_pos)
    print(dir(physicsSimulation.model))
    # print(dir(physicsSimulation.data),physicsSimulation.dataphysicsSimulation.data)

    # print(physicsSimulation.data.qpos,dir(physicsSimulation.data.qpos))
    # print(physicsSimulation.data.qpos,dir(physicsSimulation.data.qpos))
    qPosInit = [0, 0]*numAgents
    qVelInit = [0, 0]*numAgents
    qVelInitNoise = 0
    qPosInitNoise = 0.8
    getBlockRandomPos = lambda: np.random.uniform(-0.5, +0.5, 2)
    getBlockSpeed = lambda: np.zeros(2)
    reset=ResetUniformWithoutXPos(physicsSimulation, qPosInit, qVelInit, numAgents, numBlocks,qPosInitNoise, qVelInitNoise,getBlockRandomPos,getBlockSpeed)



    dt=0.1
    damping=2.5
    numSimulationFrames = 1
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


    # individStr = 'individ' if individualRewardWolf else 'shared'
    # fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}{}_agent".format(
    #     numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)


    # wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg', fileName + str(i) + '60000eps') for i in wolvesID]
    # [restoreVariables(model, path) for model, path in zip(wolvesModel, wolvesModelPaths)]
    #
    # actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    # policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    # modelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed', fileName + str(i)) for i in
    #               range(numAgents)]
    # modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG')
    # modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG','timeconst='+str(timeconst)+'dampratio='+str(dampratio))
    if hasWalls:
        modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG','hasWalls=1'+'numBlocks='+str(numBlocks)+'numSheeps='+str(numSheeps)+'numWolves='+str(numWolves)+'timeconst='+str(timeconst)+'dampratio='+str(dampratio))
        fileName = "maddpghasWalls=1{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0shared_agent".format(numWolves, numSheeps, numBlocks)

    else :
        modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG','numBlocks='+str(numBlocks)+'numSheeps='+str(numSheeps)+'numWolves='+str(numWolves)+'timeconst='+str(timeconst)+'dampratio='+str(dampratio))
        fileName = "maddpg{}wolves{}sheep{}blocks60000episodes25stepSheepSpeed1.0shared_agent".format(numWolves, numSheeps, numBlocks)
    modelPaths = [os.path.join(modelFolder,  fileName + str(i) + '60000eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


    trajList = []
    numTrajToSample = 50
    for i in range(numTrajToSample):
        np.random.seed(i)
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))

    # saveTraj
    if saveTraj:
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
        trajSavePath = os.path.join(dirName, '..', 'trajectory', trajFileName)
        saveToPickle(trajList, trajSavePath)


    # visualize
    if visualizeTraj:
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [blockColor] * numBlocks
        render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
        print(trajList[0][0],'!!!3',trajList[0][1])
        trajToRender = np.concatenate(trajList)
        render(trajToRender)


if __name__ == '__main__':
    main()
