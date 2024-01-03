from simultProblemDef import *
import matplotlib.pyplot as plt

fName = '..//data//resSucc1.pkl'
results = pd.read_pickle(fName)

deltaList = [deltas[-1] for deltas in results['deltas']]
indOpt = np.argmin(deltaList)

cOpt = results['excitations'][indOpt]
xOpt = results['positions'][indOpt]
antNorm = np.abs(results['antNorm'])

synth = problemDef(postProcess=True)
synth.farFields = results['farFields']

thetaCenters = [105]
phiCenters = [15]
thetaWidth = 3
phiWidth = 3

TP2 = TargetPattern(thetaCenters, phiCenters, thetaWidth, phiWidth, 'both')

synth.TP = TP2
synth.writeTargetPatterns()
synth.findOptimalExcitation()

synth.plotOptimal(1)
