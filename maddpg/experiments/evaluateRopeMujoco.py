import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from functionTools.loadSaveModel import saveToPickle,GetSavePath,LoadTrajectories,loadFromPickle,readParametersFromDf,restoreVariables
from functionTools.trajectory import SampleTrajectory
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividual
from environment.chasingEnv.multiAgentEnv import ResetMultiAgentChasing, ReshapeAction, RewardSheep, RewardWolf, Observe, GetCollisionForce, IntegrateState, IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState
from maddpg.maddpgAlgor.trainer.myMADDPG import *
import json
import xmltodict
import mujoco_py as mujoco
from collections import OrderedDict
from matplotlib import pyplot as plt
import itertools as it
from environment.mujocoEnv.multiAgentMujocoEnv import SampleBlockState,IsOverlap,IsTerminal
from functionTools.trajectory import ComputeStatistics#,SampleTrajectory
import pandas as pd
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
def generateSingleCondition(condition,evalNum,randomSeed):
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

        saveTraj=True
        saveImage=False
        visualizeMujoco=False
        visualizeTraj = False
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


    modelFolder = os.path.join(dirName, '..', 'trainedModels', 'mujocoMADDPGLeasedFixedEnv2','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPaths = [os.path.join(modelFolder,  fileName + str(i) + '60000eps') for i in range(numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]


    trajList = []
    # numTrajToSample = 20
    for i in range(evalNum):
        np.random.seed(randomSeed+i)
        traj = sampleTrajectory(policy)
        trajList.append(list(traj))

    # saveTraj
    if saveTraj:
        trajectorySaveExtension = '.pickle'
        fixedParameters = { 'randomSeed':randomSeed,'evalNum':evalNum}
        trajectoriesSaveDirectory=os.path.join(dirName,'..', 'trajectory','evluateRopeFixedEnv2')
        if not os.path.exists(trajectoriesSaveDirectory):
            os.makedirs(trajectoriesSaveDirectory)
        generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
        trajectorySavePath = generateTrajectorySavePath(condition)
        saveToPickle(trajList, trajectorySavePath)

        # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}stepSheepSpeed{}{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep, sheepSpeedMultiplier, individStr)
        # trajSavePath = os.path.join(dirName, '..', 'trajectory', trajFileName)
        # saveToPickle(trajList, trajSavePath)


    # visualize
    if visualizeTraj:
        demoFolder = os.path.join(dirName, '..', 'demos', 'mujocoMADDPGLeased','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

        if not os.path.exists(demoFolder):
            os.makedirs(demoFolder)
        entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters
        render = Render(entitiesSizeList, entitiesColorList, numAgents,demoFolder,saveImage, getPosFromAgentState)
        # print(trajList[0][0],'!!!3',trajList[0][1])
        trajToRender = np.concatenate(trajList)
        render(trajToRender)

def main():


    manipulatedVariables = OrderedDict()


    manipulatedVariables['damping'] = [0.0]
    manipulatedVariables['frictionloss'] = [0.0]
    manipulatedVariables['masterForce']=[0.0]

    eavluateParmetersList=['frictionloss','damping']
    lineParameter=['masterForce']
    prametersToDrop=list(set(manipulatedVariables.keys())-set(eavluateParmetersList+lineParameter))


    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    evalNum=500
    randomSeed=133




    for condition in conditions:
        print(condition)

        generateSingleCondition(condition,evalNum,randomSeed)


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    # save evaluation trajectories




    trajectoriesSaveDirectory=os.path.join(dirName,'..', 'trajectory','evluateRopeFixedEnv2')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    trajectoryExtension = '.pickle'
    trajectoryFixedParameters = { 'randomSeed':randomSeed,'evalNum':evalNum}


    getTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories


    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    def computeV(traj):
        # [print(s1[0][0]-s2[0][0],np.linalg.norm(s1[0][0][2:]-s2[0][0][2:]),np.linalg.norm(s1[0][0][:2]-s2[0][0][:2])) for s1,s2 in zip (baselineTraj,traj) ]
        vel=[np.linalg.norm(agentState[2:]) for state in traj for agentState in state[0]]
        # print (vel)
        return np.mean(vel)

    measurementFunction = computeV
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)


    darwStatisticsDf=DarwStatisticsDf(manipulatedVariables,eavluateParmetersList,lineParameter)
    subtitle='velocity'
    figIndex=0
    ylimMax=1.2
    darwStatisticsDf(statisticsDf,subtitle,figIndex,ylimMax)



    class fliterMeasurement():
        """docstring for fliterMeasurement"""
        def __init__(self, splitLength,splitMeasurement):
            self.splitLength = splitLength
            self.splitMeasurement = splitMeasurement
        def __call__(self,traj):
            [splitMeasurement(traj[i:i+self.splitLength]) for i in range(len(traj)-self.splitLength) ]

    getWolfPos=lambda state :getPosFromAgentState(state[0][0])
    getSheepfPos=lambda state :getPosFromAgentState(state[0][1])
    minDistance=0.35
    isCaught=IsTerminal(minDistance,getWolfPos,getSheepfPos)
    measurementFunction2 =lambda traj: np.mean([isCaught(state) for state in traj])

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction2)
    statisticsDf2 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf2)
    subtitle='caughtRatio(minDistance={})'.format(minDistance)
    figIndex=figIndex+1
    ylimMax=0.2
    darwStatisticsDf(statisticsDf2,subtitle,figIndex,ylimMax)

    getWolfPos=lambda state :getPosFromAgentState(state[0][0])
    getSheepfPos=lambda state :getPosFromAgentState(state[0][1])
    calculateWolfSheepDistance=lambda state:np.linalg.norm(getWolfPos(state)-getSheepfPos(state))
    measurementFunction3 =lambda traj: np.mean([calculateWolfSheepDistance(state) for state in traj])

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction3)
    statisticsDf3 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf3)
    subtitle='WolfSheepDistance'
    figIndex=figIndex+1
    ylimMax=1.6
    darwStatisticsDf(statisticsDf3,subtitle,figIndex,ylimMax)


    getWolfPos=lambda state :getPosFromAgentState(state[0][0])
    getMasterfPos=lambda state :getPosFromAgentState(state[0][2])
    calculateWolfMasterDistance=lambda state:np.linalg.norm(getWolfPos(state)-getMasterfPos(state))
    measurementFunction3 =lambda traj: np.mean([calculateWolfMasterDistance(state) for state in traj])

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction3)
    statisticsDf3 = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf3)
    subtitle='WolfMasterDistance'
    figIndex=figIndex+1
    ylimMax=0.6
    darwStatisticsDf(statisticsDf3,subtitle,figIndex,ylimMax)



    getWolfPos=lambda state :getPosFromAgentState(state[0][0])
    getSheepfPos=lambda state :getPosFromAgentState(state[0][1])

    getWolfVel=lambda state :getVelFromAgentState(state[0][0])

    def calculateCrossAngle(vel1,vel2):
        vel1complex=complex(vel1[0],vel1[1])
        vel2complex=complex(vel2[0],vel2[1])
        return np.abs(np.angle(vel2complex/vel1complex))/np.pi*180
    calculateDevation= lambda state:calculateCrossAngle(getWolfPos(state)-getSheepfPos(state),getWolfVel(state))

    measurementFunction3 =lambda traj: np.mean([calculateDevation(state) for state in traj])

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction3)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)
    subtitle='Devation'
    figIndex=figIndex+1
    darwStatisticsDf(statisticsDf,subtitle,figIndex)

    plt.show()


class DarwStatisticsDf():
    def __init__(self,manipulatedVariables,eavluateParmetersList,lineParameter):

        self.manipulatedVariables=manipulatedVariables
        self.eavluateParmetersList=eavluateParmetersList
        self.lineParameter=lineParameter


    def __call__(self, statisticsDf,subtitle,figureIndex,ylimMax=None):

        fig = plt.figure(figureIndex)
        rowName,columnName=self.eavluateParmetersList
        numRows = len(self.manipulatedVariables[rowName])
        numColumns = len(self.manipulatedVariables[columnName])
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
                if ylimMax is not None:
                    axForDraw.set_ylim(0,ylimMax)
                drawPerformanceLine(group, axForDraw,self.lineParameter)
                plotCounter += 1


        plt.suptitle(subtitle)
        plt.legend(loc='best')






def drawPerformanceLine(dataDf, axForDraw,lineParameter):

    dataDf['agentMean'] = np.array([value for value in dataDf['mean'].values])
    dataDf['agentStd'] = np.array([value for value in dataDf['std'].values])
    dataDf.plot(ax=axForDraw, y='agentMean', yerr='agentStd', marker='o')
    # for key, grp in dataDf.groupby(lineParameter):
    #     # grp.index = grp.index.droplevel(prametersToDrop)
    #     # print(grp['mean'].values)
    #     # grp['agentMean'] = np.array([value for value in grp['mean'].values])
    #     # grp['agentMean'] =  grp['mean'].values
    #     # grp['agentStd'] = np.array([value for value in grp['std'].values])
    #     # grp['agentStd'] = grp['std'].values
    #     # plt.scatter(range(len(grp['mean'].values[0])),grp['mean'].values[0],s=5,marker='x',label=lineParameter[0]+'={}'.format(key))
    #     plt.plot(grp['mean'].values[0],alpha=0.8,label=lineParameter[0]+'={}'.format(key))
    #     plt.xlabel('timeStep')

    #     # grp.plot(ax=axForDraw,x=range(5), y='mean', marker='o', label=lineParameter[0]+'={}'.format(key))
    #     # grp.scatter(ax=axForDraw, y='mean', marker='o', label=lineParameter[0]+'={}'.format(key))
    # print(dataDf)

if __name__ == '__main__':
    main()
