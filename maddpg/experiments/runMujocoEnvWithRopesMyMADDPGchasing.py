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
import math

from maddpg.maddpgAlgor.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState
from RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from functionTools.loadSaveModel import saveVariables
from environment.chasingEnv.multiAgentEnv import   RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, getPosFromAgentState, getVelFromAgentState,PunishForOutOfBound
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.mujocoEnv.multiAgentMujocoEnv import TransitionFunctionWithoutXPos,ResetUniformWithoutXPos,SampleBlockState,IsOverlap

def transferNumberListToStr(numList):
    if isinstance(numList,list):
        strList=[str(num) for num in numList]
        return ' '.join(strList)
    else:
        return str(numList)


class MakePropertyList():
    def __init__(self, transferNumberListToStr):
        self.transferNumberListToStr=transferNumberListToStr
    def __call__(self,idlist,keyNameList,valueList):
        propertyDict={}
        [propertyDict.update({geomid:{name:self.transferNumberListToStr(value) for name, value  in zip (keyNameList,values)}}) for geomid,values in zip(idlist,valueList) ]
        # print(propertyDict)
        return propertyDict




def changeJointProperty(envDict,geomPropertyDict,xmlName):
    for number,propertyDict in geomPropertyDict.items():
        # print(geomPropertyDict)
        for name,value in propertyDict.items():

            envDict['mujoco']['worldbody']['body'][number]['joint'][name][xmlName]=value

    return envDict
class ReshapeAction:
    def __init__(self,sensitivity):
        self.actionDim = 2
        self.sensitivity = sensitivity

    def __call__(self, action): # action: tuple of dim (5,1)
        # print(action)
        actionX = action[1] - action[2]
        actionY = action[3] - action[4]
        actionReshaped = np.array([actionX, actionY]) * self.sensitivity
        return actionReshaped
class TransitionFunctionWithoutXPos:
    def __init__(self, simulation,numSimulationFrames,visualize, isTerminal, reshapeActionList):
        self.simulation = simulation
        self.isTerminal = isTerminal
        self.numSimulationFrames = numSimulationFrames
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)
        self.reshapeActionList=reshapeActionList
        self.visualize=visualize
        if visualize:
            self.physicsViewer = mujoco.MjViewer(simulation)
    def __call__(self, state, actions):

        actions = [reshapeAction(action) for action,reshapeAction in zip(actions,self.reshapeActionList)]
        state = np.asarray(state)
        # print("state", state)
        actions = np.asarray(actions)
        numAgent = len(state)
        # print(state,actions)
        oldQPos = state[:, 0:self.numJointEachSite].flatten()
        oldQVel = state[:, -self.numJointEachSite:].flatten()
        # print(self.simulation.data.qpos)
        # print(self.simulation.data.ctrl)
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = actions.flatten()
        for simulationFrame in range(self.numSimulationFrames):
            self.simulation.step()
            self.simulation.forward()
            if self.visualize:
                self.physicsViewer.render()
            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel

            agentNewQPos = lambda agentIndex: newQPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
            agentNewQVel = lambda agentIndex: newQVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
            agentNewState = lambda agentIndex: np.concatenate([agentNewQPos(agentIndex), agentNewQVel(agentIndex)])
            newState = np.asarray([agentNewState(agentIndex) for agentIndex in range(numAgent)])

            if self.isTerminal(newState):
                break

        return newState
class ResetUniformWithoutXPosForLeashed:
    def __init__(self, simulation, qPosInit, qVelInit, numAgent, tiedAgentIndex, ropePartIndex, maxRopePartLength, qPosInitNoise=0, qVelInitNoise=0):
        self.simulation = simulation
        self.qPosInit = np.asarray(qPosInit)
        self.qVelInit = np.asarray(qVelInit)
        self.numAgent = self.simulation.model.nsite
        self.tiedBasePosAgentIndex, self.tiedFollowPosAgentIndex = tiedAgentIndex
        self.numRopePart = len(ropePartIndex)
        self.maxRopePartLength = maxRopePartLength
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
        self.numJointEachSite = int(self.simulation.model.njnt/self.simulation.model.nsite)

    def __call__(self):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        # print(numQPos)
        qPos = self.qPosInit + np.random.uniform(low=-self.qPosInitNoise, high=self.qPosInitNoise, size=numQPos)
        tiedBasePos = qPos[self.numJointEachSite * self.tiedBasePosAgentIndex: self.numJointEachSite * (self.tiedBasePosAgentIndex + 1)]
        sampledRopeLength = np.random.uniform(low = 0, high = self.numRopePart * self.maxRopePartLength)
        sampledPartLength = np.arange(sampledRopeLength/(self.numRopePart + 1), sampledRopeLength, sampledRopeLength/(self.numRopePart + 1))[:self.numRopePart]
        theta = np.random.uniform(low = 0, high = math.pi)

        tiedFollowPosAgentPos = tiedBasePos + np.array([sampledRopeLength * np.cos(theta), sampledRopeLength * np.sin(theta)])
        qPos[self.numJointEachSite * self.tiedFollowPosAgentIndex : self.numJointEachSite * (self.tiedFollowPosAgentIndex + 1)] = tiedFollowPosAgentPos
        ropePartPos = np.array(list(zip(sampledPartLength * np.cos(theta), sampledPartLength * np.sin(theta)))) + tiedBasePos
        qPos[-self.numJointEachSite * self.numRopePart : ] = np.concatenate(ropePartPos)

        qVelSampled = np.concatenate([np.random.uniform(low=-self.qVelInitNoise, high=self.qVelInitNoise, size=numQVel - self.numRopePart * self.numJointEachSite),\
                                      np.zeros(self.numRopePart * self.numJointEachSite)])
        qVel = self.qVelInit + qVelSampled

        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()

        agentQPos = lambda agentIndex: qPos[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentQVel = lambda agentIndex: qVel[self.numJointEachSite * agentIndex : self.numJointEachSite * (agentIndex + 1)]
        agentState = lambda agentIndex: np.concatenate([agentQPos(agentIndex), agentQVel(agentIndex)])
        startState = np.asarray([agentState(agentIndex) for agentIndex in range(self.numAgent)])

        return startState
# fixed training parameters
maxEpisode = 60000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numMasters saveAllmodels = True or False

def main():
    debug = 0
    if debug:

        damping=0.0
        frictionloss=0.4
        masterForce=1.0


        numWolves = 1
        numSheeps = 1
        numMasters = 1
        saveAllmodels = True
        maxTimeStep = 25
        visualize=False

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])

        maxTimeStep = 25
        visualize=False
        saveAllmodels = True
    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps,  save all models: {}".
          format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep,  str(saveAllmodels)))
    print(damping,frictionloss,masterForce)

    modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPGLeasedFixedEnv2','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)


    numAgents = numWolves + numSheeps+numMasters
    numEntities = numAgents + numMasters
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.075
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numMasters



    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound)


    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))+list(rewardMaster(state, action, nextState))


    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3]
    keyNameList=[0,1]
    valueList=[[frictionloss,frictionloss]]*len(geomIds)
    frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')


    physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','rope','leasedNew.xml')
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()


    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)



    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)


    qPosInit = (0, ) * 24
    qVelInit = (0, ) * 24
    qPosInitNoise = 0.6
    qVelInitNoise = 0
    numAgent = 3
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(3, 12))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

    numSimulationFrames=10
    isTerminal= lambda state: False
    reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce)]
    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualize,isTerminal, reshapeActionList)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, masterID, getPosFromAgentState,getVelFromAgentState)
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
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPath = os.path.join(modelFolder, fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath+ str(i), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgents)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = maddpg(replayBuffer)





if __name__ == '__main__':
    main()


