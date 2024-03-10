from src.antennaArrays.ArrayPostProcessing import ArrayPostProcessing
from src.antennaArrays.AntennaArray import TargetPattern

archivePath = '..//data//resSimult1_20240310_180919.pkl'

arrayPostProc = ArrayPostProcessing(archivePath)


thetaCenters = [105]
phiCenters = [15]
thetaWidth = 3
phiWidth = 3

TP2 = TargetPattern(thetaCenters, phiCenters, thetaWidth, phiWidth, 'both')

arrayPostProc.TP = TP2
arrayPostProc.writeTargetPatterns()
arrayPostProc.findOptimalExcitation()

arrayPostProc.plotOptimal(1)
