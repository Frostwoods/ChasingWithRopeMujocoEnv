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
from collections import OrderedDict
import itertools as it
from environment.mujocoEnv.multiAgentMujocoEnv import SampleBlockState,IsOverlap


wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])

maxEpisode = 60000
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
    def __init__(self, simulation,numSimulationFrames, visualize,isTerminal, reshapeActionList):
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
        # print( self.numRopePart ,self.maxRopePartLength)
        # print(sampledRopeLength)
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
        # print(startState[0:6])
        return startState
def generateSingleCondition(condition):
    debug = 0
    if debug:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        maxTimeStep = 25

        saveTraj=False
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True
    else:

        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        maxTimeStep = 25

        saveTraj=False
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=False

    print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))


    numAgents = numWolves + numSheeps+numMasters
    numEntities = numAgents + numMasters
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]

    wolfSize = 0.075
    sheepSize = 0.05
    masterSize = 0.075
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None


    entitiesMovableList = [True] * numAgents + [False] * numMasters
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)


    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))+list(rewardMaster(state, action, nextState))

    dirName = os.path.dirname(__file__)
    # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','rope','leased.xml')

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
    # physicsDynamicsPath=os.path.join(dirName,'..','..','environment','mujocoEnv','rope','leased.xml')


    with open(physicsDynamicsPath) as f:
        xml_string = f.read()
    envXmlDict = xmltodict.parse(xml_string.strip())




    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)





    envXml=xmltodict.unparse(envXmlDict)

    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)
    # print(physicsSimulation.model.body_pos)

    # print(dir(physicsSimulation.model))
    # print(dir(physicsSimulation.data),physicsSimulation.dataphysicsSimulation.data)

    # print(physicsSimulation.data.qpos,dir(physicsSimulation.data.qpos))
    # print(physicsSimulation.data.qpos,dir(physicsSimulation.data.qpos))
    qPosInit = (0, ) * 24
    qVelInit = (0, ) * 24
    qPosInitNoise = 0.4
    qVelInitNoise = 0
    numAgent = 2
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(3, 12))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    numSimulationFrames=10
    isTerminal= lambda state: False
    reshapeActionList = [ReshapeAction(5),ReshapeAction(5),ReshapeAction(masterForce)]
    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)

    # damping=2.5
    # numSimulationFrames =int(0.1/dt)
    # agentsMaxSpeedList = [wolfMaxSpeed]* numWolves + [sheepMaxSpeed] * numSheeps
    # reshapeAction = ReshapeAction()
    # isTerminal = lambda state: False
    # transit = TransitionFunctionWithoutXPos(physicsSimulation,numAgents , numSimulationFrames,damping*dt*numSimulationFrames,agentsMaxSpeedList,visualizeMujoco,isTerminal,reshapeAction)


    maxRunningStepsToSample = 100
    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, masterID, getPosFromAgentState, getVelFromAgentState)
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
    #     numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)


    # wolvesModelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg', fileName + str(i) + '60000eps') for i in wolvesID]
    # [restoreVariables(model, path) for model, path in zip(wolvesModel, wolvesModelPaths)]
    #
    # actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    # policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    # modelPaths = [os.path.join(dirName, '..', 'trainedModels', '3wolvesMaddpg_ExpEpsLengthAndSheepSpeed', fileName + str(i)) for i in
    #               range(numAgents)]
    # modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG')
    # modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPG','timeconst='+str(timeconst)+'dampratio='+str(dampratio))

    modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPGLeasedFixedEnv2','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) + '60000eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


    trajList = []
    numTrajToSample = 5
    for i in range(numTrajToSample):
        np.random.seed(i)
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))

    # saveTraj
    if saveTraj:
        trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
        trajSavePath = os.path.join(dirName, '..', 'trajectory', trajFileName)
        saveToPickle(trajList, trajSavePath)


    # visualize
    if visualizeTraj:
        demoFolder = os.path.join(dirName, '..', 'demos', 'mujocoMADDPGLeasedFixedEnv2','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

        if not os.path.exists(demoFolder):
            os.makedirs(demoFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters
        render = Render(entitiesSizeList, entitiesColorList, numAgents,demoFolder,saveImage, getPosFromAgentState)
        # print(trajList[0][0],'!!!3',trajList[0][1])
        trajToRender = np.concatenate(trajList)
        render(trajToRender)
    # if makeVideo:
    #     import cv2

    #     videoFolder=os.path.join(dirName, '..', 'demos', 'video')
    #     if not os.path.exists(videoFolder):
    #         os.makedirs(videoFolder)

    #     videoPath= os.path.join(videoFolder,'mujocoMADDPGLeased_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    #     fps = 20
    #     size=(700,700)
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     videoWriter = cv2.VideoWriter(videoPath,fourcc,fps,size)#最后一个是保存图片的尺寸

    #     #for(i=1;i<471;++i)
    #     for i in range(0,numTrajToSample*maxRunningStepsToSample):
    #         imgPath=os.path.join(demoFolder,'rope'+str(i)+'.png')
    #         frame = cv2.imread(imgPath)
    #         videoWriter.write(frame)
    #     videoWriter.release()
def main():


    manipulatedVariables = OrderedDict()


    manipulatedVariables['damping'] = [0.0,1.0]
    manipulatedVariables['frictionloss'] = [0.0,0.2,0.4]
    manipulatedVariables['masterForce']=[0.0,2.0]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    for condition in conditions:
        print(condition)
        try :
            generateSingleCondition(condition)
        except :
            continue
if __name__ == '__main__':
    main()
