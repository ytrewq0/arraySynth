from src.antennaArrays.ArraySynthesis import ArraySynthesis
from src.antennaArrays.AntennaArray import BasicData, TargetPattern, OptimizationParams
import numpy as np

angStep = 1
fileName = 'patch_circ_dual'
processNum = 1

BD = BasicData(angStep, fileName, processNum)

thetaCenters = [80, 90]
phiCenters = [0, 10]
thetaWidth = 10
phiWidth = 10

TP = TargetPattern(thetaCenters, phiCenters, thetaWidth, phiWidth, 'both')

N = 9
runTime = 600
xMin = np.array([80, 80])
xMax = np.array([150, 150])
x0 = np.array([100, 100])
geomParams = ['xd0', 'yd0']
ampSymbol = 'c'
phaseSymbol = 'p'

OP = OptimizationParams(N, runTime, xMin, xMax, x0, geomParams, ampSymbol, phaseSymbol, dualPort=True)

synth = ArraySynthesis(TP, OP, BD)
synth.simultaneousOptimizer()

