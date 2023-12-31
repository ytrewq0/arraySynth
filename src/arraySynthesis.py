import numpy as np
import os
import sys
import copy
import nlopt
import pandas as pd
from pyaedt import *
import time
import shutil
import pickle
import matplotlib.pyplot as plt

# degree to radian conversion
aconst = 1 / 180 * np.pi



class TargetPattern:
    """
    Class describing the patterns to synthesize
    """

    def __init__(self, thetaCenters, phiCenters, thetaWidth, phiWidth, polarization, angleMargin=0.1):
        """
        :param thetaCenters [degree]: lobe centers theta direction:
        :param phiCenters [degree]: lobe centers phi direction
        :param thetaWidth [degree]: lobe width theta direction
        :param phiWidth [degree]:  lobe width phi direction
        :param polarization: 'theta', 'phi' or 'both'
        :param angleMargin: numerical padding for angular condition checks (alpha < beta)
        """
        self.thetaCenters = thetaCenters
        self.phiCenters = phiCenters
        self.thetaWidth = thetaWidth
        self.phiWidth = phiWidth
        self.polarization = polarization
        self.angleMargin = angleMargin

        # Calculate lobe boundaries
        self.calculateLobeBoundaries()

        # Calculate direction vectors based on polarization
        self.calculateDirectionVectors()

        # Number of goal patterns / directions
        self.goalNum = self.lims.shape[0]
        self.thetaLims = self.lims[:, 0]
        self.phiLims = self.lims[:, 1]

    def calculateLobeBoundaries(self):
        """
        specify the angle interval characterizing the lobe
        """
        lims = []
        for theta in self.thetaCenters:
            for phi in self.phiCenters:
                lims.append(self.calculateSingleLobe(theta, phi))
        if self.polarization == 'both':
            self.lims = np.array(lims + lims)
        else:
            self.lims = np.array(lims)

    def calculateSingleLobe(self, theta, phi):
        """
        get a single lobe interval from center point and lobe width
        :param theta: theta lobe center
        :param phi: phi lobe center
        :return: lobe interval
        """
        return (
            [theta - self.thetaWidth / 2, theta + self.thetaWidth / 2],
            [phi - self.phiWidth / 2, phi + self.phiWidth / 2]
        )

    def calculateDirectionVectors(self):
        """
        get array of direction vectors based on polarization requirements
        """
        if self.polarization == 'theta':
            self.dirVec = np.array([[1, 0]] * len(self.lims))
        elif self.polarization == 'phi':
            self.dirVec = np.array([[0, 1]] * len(self.lims))
        elif self.polarization == 'both':
            self.dirVec = np.array([[1, 0]] * (len(self.lims) // 2) + [[0, 1]] * (len(self.lims) // 2))
        else:
            raise ValueError("polarization should be 'theta', 'phi' or 'both'")


class OptimizationParams:
    """
    Parameters for the optimizer
    """

    def __init__(self, N, runTime, xMin, xMax, x0, geomParamNames, ampSymbol = 'c', phaseSymbol = 'p',
                 method=nlopt.GN_DIRECT_L_RAND, dualPort=False):
        """
        :param N: number of antennas
        :param runTime: optimization time limit in seconds
        :param xMin: array of minimum limit for the optimization parameters
        :param xMax: -||- maximum
        :param x0: initial step of the optimization (may get overwritten in the successive algorithm)
        :param geomParamNames: names of the optimizable geometric params in the hfss model
        :param ampSymbol: symbol for the amplitudes in the hffs model
        :param phase: symbol for the phases in the hffs model
        :param method: optimization method
        :param dualPort: whether there's two ports on the antenna
        """
        self.N = N
        self.dualPort = dualPort
        self.runTime = runTime

        assert (len(xMin) == len(xMax) == len(x0) == len(geomParamNames))
        self.xMin = xMin
        self.xMax = xMax
        self.x0 = x0
        self.geomParamNames = geomParamNames
        self.ampSymbol = ampSymbol
        self.phaseSymbol = phaseSymbol
        self.optimParamNum = len(xMin)
        self.method = method


class BasicData:
    """
        Basic input data for the synthesis
    """

    def __init__(self, angStep, fileName, processNum=1, maxPasses=None):
        """
        :param angStep: angular resolution of the output far fields
        :param fileName: obvious
        :param processNum: process number (relevant for parallel runs), ignore for now
        :param maxPasses: maximum number of passes in the adaptive HFSS solver
        
        """
        # Anglular resolution of output
        self.angStep = angStep
        # Impedance of free space
        self.Z0 = 376.73
        # source code directory
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        # process number (relevant for parallel runs)
        self.processNum = processNum
        # path of the .aedt file to use
        self.designPath = (self.dir_path + f'\\{fileName}{processNum}.aedt')
        # units
        self.unitDict = {'uV': 1e-6, 'mV': 1e-3, 'V': 1}
        # field components
        self.components = ['__theta', '__phi']
        # number of max passes (HFSS parameter)
        self.maxPasses = maxPasses


class Synthesizer:
    """
        Object realizing the synthesis
    """

    def __init__(self, TP, OP, BD):
        """
        :param TP, OP, BD: target, optimization and basic data objects
        """
        self.TP = TP
        self.OP = OP
        self.BD = BD

        # list of the difference between goal and realized patterns in each iteration; last value is the L2 norm
        self.deltas = []
        # list of positions in each iteration
        self.positions = []
        # list of excitations in each iteration
        self.excitations = []
        # list of simulation runtimes
        self.simTimes = []
        # successive method: all positons
        self.allX = []
        # container for the optimal excitation of the current geometry
        self.optimalExcitation = []
        # container for the synthesis errors of the current iteration
        self.currentDeltas = np.zeros(self.TP.goalNum + 1)
        # simulation runTime for the current simulation
        self.simTime = None

        if self.OP.dualPort == True:
            # pattern norm vector
            self.antNorm = np.empty(2 * self.OP.N, np.complex128)
            # mutual inductance matrix
            self.mutY = np.empty((2 * self.OP.N, 2 * self.OP.N), np.complex128)
            # mixed term RHS
            self.mixedTerm = np.empty((self.TP.goalNum, 2 * self.OP.N), np.complex128)
        else:
            self.antNorm = np.empty(self.OP.N, np.complex128)
            self.mutY = np.empty((self.OP.N, self.OP.N), np.complex128)
            self.mixedTerm = np.empty((self.TP.goalNum, self.OP.N), np.complex128)

        # target norm vector
        self.targetNorm = np.empty(self.TP.goalNum, np.complex128)
        # impedance matrix placeholder
        self.mutZ = None
        
        # port string keys ('' for normal, '_p' for second (perpendicular) port
        if self.OP.dualPort:
            self.ports = ['', '_p']
        else:
            self.ports = ['']
        
        # dict for transforming port descriptor to index
        self.portKey = {'': 0, "_p": self.OP.N}

        # HFSS  handle
        self.app = None
        # far field dataframe
        self.farFields = None
        
    def fastGenerate(self):
        """
        Launch HFSS, generate the far fields dataframe containing the fields corresponding to unit excitations and the 
        target fields and prepare for synthesis run
        """
        self.app = Hfss()
        self.app.load_project(self.BD.designPath)
        self.app.field_setups[0].phi_step = self.BD.angStep
        self.app.field_setups[0].theta_step = self.BD.angStep

        setupName = self.app.setups[0].name
        maxPassesOriginal = self.app.setups[0].props['MaximumPasses']
        self.app.edit_setup(setupName, {'MaximumPasses': 1})
        self.generateFarFieldDF()
        if self.BD.maxPasses is None:
            self.app.edit_setup(setupName, {'MaximumPasses': maxPassesOriginal})
        else:
            self.app.edit_setup(setupName, {'MaximumPasses': self.BD.maxPasses})
        self.app.close_project()

    def generateFarFieldDF(self):
        """
        set up the dataframe containing all far field data
        """
        self.app.analyze()
        self.app.settings.enable_pandas_output = True

        # get far field values from API
        vals = self.app.post.get_far_field_data(["rEPhi", "rETheta"])
        self.farFields = vals.full_matrix_real_imag
        self.farFields = self.farFields[0] + 1j * self.farFields[1]

        # Data format is: for cn excitation, [real(E_theta) + imag(E_theta), real(E_phi) + imag(E_phi)]
        for ind in range(1, self.OP.N + 1):
            for port in self.ports:
                self.farFields[f'c{port}{ind}__theta'] = copy.deepcopy(self.farFields['rETheta'])
                self.farFields[f'c{port}{ind}__phi'] = copy.deepcopy(self.farFields['rEPhi'])

        # delete clutter
        self.farFields = self.farFields.drop(['rETheta', 'rEPhi'], axis=1)

        self.intCoeff()
        self.writeTargetPatterns()
        self.hfssSetZeroExcit()

    def intCoeff(self):
        """
        add the integration coefficients to the main dataframe
        """
        # assign names to the multiindex
        self.farFields.index = self.farFields.index.set_names(['freq', 'phi', 'theta'])

        # Define conditions for 'intCoeff' using np.isclose (integral correction at 'doubly' counted points )
        thetaCond = (np.isclose(self.farFields.index.get_level_values('theta'), 0, atol=0.01) +
                     np.isclose(self.farFields.index.get_level_values('theta'), 180, atol=0.01))

        phiCond = (np.isclose(self.farFields.index.get_level_values('phi'), -180, atol=0.01) +
                   np.isclose(self.farFields.index.get_level_values('phi'), 180, atol=0.01))

        # Assign values to 'intCoeff' based on conditions
        self.farFields['intCoeff'] = 0.5 * (thetaCond | phiCond) + 1 * ~(thetaCond | phiCond)

    def writeTargetPatterns(self):
        """
        calculate and save all goal patterns to the main dataframe
        """
        # Add goal patterns to the output:
        for goal in range(1, self.TP.goalNum + 1):
            for ind, component in enumerate(self.BD.components):
                self.farFields[f'goal{goal}{component}'] = self.farFields[f'c1{component}'] * 0

                condition = (
                        (self.farFields.index.get_level_values('phi') >= self.TP.phiLims[goal - 1][
                            0] - self.TP.angleMargin) &
                        (self.farFields.index.get_level_values('phi') - self.TP.angleMargin <=
                         self.TP.phiLims[goal - 1][1]) &
                        (self.farFields.index.get_level_values('theta') + self.TP.angleMargin >=
                         self.TP.thetaLims[goal - 1][0]) &
                        (self.farFields.index.get_level_values('theta') - self.TP.angleMargin <=
                         self.TP.thetaLims[goal - 1][1])
                )

                self.farFields.loc[condition, f'goal{goal}{component}'] = self.TP.dirVec[goal - 1][ind]
        self.normalizeTargetPatterns()

    def normalizeTargetPatterns(self):
        """
        self-explanatory
        """
        self.integrateTarget()
        for goal in range(1, self.TP.goalNum + 1):
            for component in self.BD.components:
                self.farFields[f'goal{goal}{component}'] = self.farFields[f'goal{goal}{component}'] / np.sqrt(
                    self.targetNorm[goal - 1])

    def hfssSetZeroExcit(self):
        """
        set zero values for the excitations. Also serves as a check to see whether they are defined in the model
        """
        for j in range(1, self.OP.N + 1):
            self.app[f'{self.OP.ampSymbol}{j}'] = '0.000001V'
            self.app[f'{self.OP.ampSymbol}_p{j}'] = '0.000001V'
            self.app[f'{self.OP.phaseSymbol}{j}'] = '0.0'
            self.app[f'{self.OP.phaseSymbol}_p{j}'] = '0.0'

        thStep = self.farFields.index.get_level_values('theta').unique()[1] - \
                 self.farFields.index.get_level_values('theta').unique()[0]
        phStep = self.farFields.index.get_level_values('phi').unique()[1] - \
                 self.farFields.index.get_level_values('phi').unique()[0]
        assert np.abs(thStep - phStep) / thStep < 0.01

    def innerProduct(self, var1, var2):
        """
        inner product operator definition
        :param var1: operand, dataframe column
        :param var2: same
        """
        return self.BD.angStep ** 2 * aconst ** 2 * np.sum(
            self.farFields['intCoeff'] * np.sin(self.farFields.index.get_level_values('theta') * aconst) *
            np.conjugate(var2) * var1)

    def integrateAntFields(self):
        """
        calculate norms of antenna excitations, and normalize fields with them
        """
        self.antNorm = np.zeros_like(self.antNorm)
        for ant in range(1, self.OP.N + 1):
            for port in self.ports:
                for component in self.BD.components:
                    self.antNorm[ant - 1 + self.portKey[port]] += self.innerProduct(
                        self.farFields[f'c{port}{ant}{component}'],
                        self.farFields[f'c{port}{ant}{component}'])

        self.normalizeAntFields()

    def normalizeAntFields(self):
        """
        self-explanatory
        """
        for ant in range(1, self.OP.N):
            for port in self.ports:
                for component in self.BD.components:
                    self.farFields[f'c{port}{ant}{component}'] = \
                        self.farFields[f'c{port}{ant}{component}'] / np.sqrt(
                            np.real(self.antNorm)[ant + self.portKey[port] - 1])

    def integrateMutual(self):
        """
        function to calculate inner product between the fields corresponding to different excitations
        """

        self.mutY = np.zeros_like(self.mutY)
        self.mutZ = np.zeros_like(self.mutZ)

        for ant1 in range(1, self.OP.N + 1):
            for ant2 in range(1, self.OP.N + 1):
                for port1 in self.ports:
                    for port2 in self.ports:
                        for component in self.BD.components:
                            self.mutY[ant1 + self.portKey[port1] - 1, ant2 + self.portKey[port2] - 1] += \
                                self.innerProduct(self.farFields[f'c{port2}{ant2}{component}'],
                                                  self.farFields[f'c{port1}{ant1}{component}'])

        self.mutZ = np.linalg.inv(self.mutY)

    def integrateMixed(self):
        """
        function to calculate inner product between the fields corresponding to an excitations and the target pattern
        """
        self.mixedTerm = np.zeros_like(self.mixedTerm)

        for goal in range(1, self.TP.goalNum + 1):
            for ant in range(1, self.OP.N + 1):
                for port in self.ports:
                    for component in self.BD.components:
                        self.mixedTerm[goal - 1, ant - 1 + self.portKey[port]] += \
                            self.innerProduct(self.farFields[f'goal{goal}{component}'],
                                              self.farFields[f'c{port}{ant}{component}'])

    def integrateTarget(self):
        """
        function to calculate the 'norm' of the target pattern
        """
        self.targetNorm = np.zeros_like(self.targetNorm)

        for goal in range(1, self.TP.goalNum + 1):
            for component in self.BD.components:
                self.targetNorm[goal - 1] += self.innerProduct(self.farFields[f'goal{goal}{component}'],
                                                               self.farFields[f'goal{goal}{component}'])

    def fieldFromExcit(self, excitation):
        """
        :param excitation: vector of excitation voltages
        :return: radiated far field [theta, phi] components
        """
        synthField = [0 * self.farFields['c1__theta'], 0 * self.farFields['c1__phi']]
        for ant in range(1, self.OP.N + 1):
            for port in self.ports:
                for ind, component in enumerate(self.BD.components):
                    synthField[ind] += excitation[ant - 1 + self.portKey[port]] * self.farFields[
                        f'c{port}{ant}{component}']
        return synthField

    def integrateByDef(self, excitations):
        """
            function to calculate the norm of the difference between a synthesized field weighed by the excitation
            coefficients and the target pattern
            :param excitations: vector of vectors containing the excitations of the individual antennas
            :return delt: norm of the difference
        """
        delt = np.zeros(self.TP.goalNum)
        for goal in range(1, self.TP.goalNum + 1):
            synthField = self.fieldFromExcit(excitations[goal - 1])
            for ind, component in enumerate(self.BD.components):
                delt[goal - 1] += self.innerProduct(synthField[ind] - self.farFields[f'goal{goal}{component}'],
                                                    synthField[ind] - self.farFields[f'goal{goal}{component}'])

        return delt

    def setAnalyze(self, geomParams):
        """
        set geometry and solve setup in HFSS
        :param geomParams: geometric parameters
        """
        self.app.load_project(self.BD.designPath)
        for ind, param in enumerate(geomParams):
            self.app[self.OP.geomParamNames[ind]] = f'{param}mm'
        self.app.analyze()

    def readFields(self):
        """
        read the fields after solution to dataframe 'self.farFields'
        """
        # write far field values to dataframes
        for ind in range(1, self.OP.N + 1):
            for port in self.ports:
                print(f'reading config {self.OP.ampSymbol}{port}{ind}')
                self.app[f'{self.OP.ampSymbol}{port}{ind}'] = '1V'
                # get far field values from API
                vals = self.app.post.get_far_field_data(["rEPhi", "rETheta"])
                fields = vals.full_matrix_real_imag
                fields = fields[0] + 1j * fields[1]

                # normalize to V
                # Data format is: for cn excitation, [real(E_theta) + imag(E_theta), real(E_phi) + imag(E_phi)]
                self.farFields[f'c{port}{ind}__theta'] = self.BD.unitDict[vals.units_data['rETheta']] * fields[
                    'rETheta']
                self.farFields[f'c{port}{ind}__phi'] = self.BD.unitDict[vals.units_data['rEPhi']] * fields['rEPhi']

                self.app[f'{self.OP.ampSymbol}{port}{ind}'] = '0.000001V'

    def cleanup(self):
        """
        cleanup routine; close project and delete results
        """
        self.app.close_project()
        # delete previous calc results to prevent save time snowballing
        projectFolder = (self.BD.designPath.split('.'))[0]
        projectFolder = projectFolder + '.aedtresults'
        if os.path.exists(projectFolder) and os.path.isdir(projectFolder):
            shutil.rmtree(projectFolder)

    def recalc(self, geomParams):
        """
        change input parameters of HFSS model, run the simulation and write the results to self.farFields
        """
        self.setAnalyze(geomParams)
        self.readFields()
        self.cleanup()

    def simulateAndFetch(self, geomParams):
        """
        feed new geometry to HFSS model, run simulation and extract data
        :return:
        """
        # timer
        t1 = time.perf_counter()

        # call the API to calculate the far fields
        self.recalc(geomParams)
        self.simTime = time.perf_counter() - t1

    def findOptimalExcitation(self):
        """
        find the optimal excitation to achieve the desired goal patterns, refer to the formulas in the paper by Marak et al.
        """
        self.integrateAntFields()
        self.integrateMutual()
        self.integrateMixed()

        deltaAux = np.zeros(self.TP.goalNum + 1)
        self.currentDeltas = np.zeros(self.TP.goalNum + 1)
        self.optimalExcitation = []

        for goal in range(0, self.TP.goalNum):
            deltaAux[goal] = np.abs(
                (np.dot(self.mixedTerm[goal, :].T.conjugate(), np.dot(self.mutZ, self.mixedTerm[goal, :]))))
            self.optimalExcitation.append((np.dot(self.mutZ, self.mixedTerm[goal, :])))

        self.currentDeltas[0:-1] = 1 - deltaAux[0:-1]
        self.currentDeltas[-1] = np.linalg.norm(self.currentDeltas[0: -1]) / np.sqrt(self.TP.goalNum)

        d0 = self.integrateByDef(self.optimalExcitation)

    def writeLog(self, geomParams):
        """
        Write iteration results into text files and append to the class' tracking lists
        :param geomParams: geometric parameters
        """
        dataFolderExists = os.path.exists('../data')
        if not dataFolderExists:
            # Create a new directory because it does not exist
            os.makedirs('../data')
            print("Data folder created!")
        else:
            print("Data folder already exists, using it to store the data")

        with open('data/deltas' + str(self.BD.processNum) + '.txt', 'a') as f:
            f.write(str(self.currentDeltas) + '        ; ' + 'simulation runtime = ' + str(self.simTime) + '\n')
            f.write('\n')
        with open('data/positions' + str(self.BD.processNum) + '.txt', 'a') as f:
            f.write(str(geomParams) + '\n')
            f.write('\n')
        with open('data/optimalExcitation' + str(self.BD.processNum) + '.txt', 'a') as f:
            f.write(str(self.optimalExcitation) + '\n')
            f.write('\n')

        self.deltas.append(copy.deepcopy(self.currentDeltas))
        self.excitations.append(copy.deepcopy(self.optimalExcitation))
        self.positions.append(copy.deepcopy(geomParams))
        self.simTimes.append(copy.deepcopy(self.simTime))

    def synthStep(self, geomParams, gradient):
        """
        synthesis step, perform optimization for selected parameters on the input
        :param geomParams: geometric parameters
        """
        self.simulateAndFetch(geomParams)
        self.findOptimalExcitation()
        self.writeLog(geomParams)
        return self.currentDeltas[-1]

    def pickleSaver(self, archiveName):
        """
        save important fields to a pickle archive
        :param archiveName: name-code for the archive
        """
        fieldsToSave = {'deltas': self.deltas, 'excitations': self.excitations, 'positions': self.positions,
                        'runTimes': self.simTimes, 'allX': self.allX, 'x0': self.OP.x0, 'antNorm': self.antNorm,
                        'farFields': self.farFields}

        # Save the selected fields to a pickle archive
        with open(f'data/res{archiveName}{self.BD.processNum}.pkl', 'wb') as file:
            pickle.dump(fieldsToSave, file)

    def simultaneousOptimizer(self):
        """
        simultaneous optimizer definition
        """
        # instantiate optimizer object
        opt = nlopt.opt(self.OP.method, self.OP.optimParamNum)

        # set function to optimize
        opt.set_min_objective(self.synthStep)

        opt.set_lower_bounds(self.OP.xMin)
        opt.set_upper_bounds(self.OP.xMax)
        # time set for optimization (we use a time limit because characterizing the convergence of global optimization is difficult)
        opt.set_maxtime(self.OP.runTime)
        # run the optimization
        self.OP.x0 = opt.optimize(self.OP.x0)

        self.recalc(self.OP.x0)
        self.app.close_desktop()

        self.pickleSaver('Simult')

    def successiveOptimizer(self, outerIters):
        """
        successive optimizer definition
        :param outerIters: number of 'outer' iterations, meaning the number of antenna replacements in an optimization
        run
        """
        for iter in range(0, outerIters):
            for optimIndex, optimParam in enumerate(self.OP.geomParamNames):
                # instantiate optimizer object
                opt = nlopt.opt(self.OP.method, 1)
                # set function to optimize
                opt.set_min_objective(self.synthStep)

                opt.set_lower_bounds([self.OP.xMin[optimIndex]])
                opt.set_upper_bounds([self.OP.xMax[optimIndex]])
                # time set for optimization (we use a time limit because characterizing the convergence of global optimization is difficult)
                opt.set_maxtime(self.OP.runTime)
                # run the optimization
                self.OP.x0[optimIndex] = opt.optimize([self.OP.x0[optimIndex]])
                self.allX.append(copy.deepcopy(self.OP.x0))

        self.recalc(self.OP.x0)
        self.app.close_desktop()

        self.pickleSaver('Succ')

    def angleGrid(self):
        """
        get theta and phi multiindex values from a far field dataframe
        """
        # Extract levels for phi and theta coordinates
        phiVals = self.farFields.index.get_level_values('phi')
        thetaVals = self.farFields.index.get_level_values('theta')
        return thetaVals, phiVals

    def plotPreprocessing(self, field, maxTheta=None, maxPhi=None):
        """
        find maximum values of the given field in terms of spherical angles theta and phi
        :param field: a column of the far field dataframe
        :param maxTheta: manual input for the maximum, overrides the maximization routine
        :param maxPhi: analogous
        """
        thetaVals, phiVals = self.angleGrid()
        # Find the indices of the maximum value in the synthetic field
        maxIndTheta = np.argmax(np.abs(field[0]))
        maxIndPhi = np.argmax(np.abs(field[1]))

        if maxTheta:
            eThetaMax__Theta = maxTheta
            ePhiMax__Theta = maxTheta
        else:
            eThetaMax__Theta = thetaVals[maxIndTheta]
            ePhiMax__Theta = thetaVals[maxIndPhi]

        if maxPhi:
            eThetaMax__Phi = maxPhi
            ePhiMax__Phi = maxPhi
        else:
            eThetaMax__Phi = phiVals[maxIndTheta]
            ePhiMax__Phi = phiVals[maxIndPhi]

        return eThetaMax__Theta, eThetaMax__Phi, ePhiMax__Theta, ePhiMax__Phi

    def plotBase(self, fieldComponent, maxTheta, maxPhi, title, addTarget=False, showDelta=False):
        """
        plotting function
        :param fieldComponent: column of the far field dataframe
        :param maxTheta: the coordinate of fieldComponent's maximum absolute value
        :param maxPhi: analogous
        :param title: plot title
        :param addTarget: target pattern limits
        :param showDelta: delta (synthesis error value)
        """
        thetaVals, phiVals = self.angleGrid()
        # Create a scatter plot with color-coded complex values
        plt.figure(figsize=(8, 6))
        contour = plt.tricontourf(thetaVals, phiVals, np.abs(fieldComponent), cmap='viridis', levels=50)
        plt.colorbar(contour, label='Absolute Value')
        plt.xlabel('phi')
        plt.ylabel('theta')
        plt.title(title)

        # Add dashed lines passing through the maximum point
        plt.axvline(x=maxTheta, linestyle='--', color='white', linewidth=1)
        plt.axhline(y=maxPhi, linestyle='--', color='white', linewidth=1)

        # Add labels for intersections with the x and y axes
        plt.text(maxTheta, phiVals.min() - 0.1, f'{maxTheta:.2f}', color='white', ha='center')
        plt.text(thetaVals.min() - 0.1, maxPhi, f'{maxPhi:.2f}', color='white', va='center',
                 rotation='vertical')

        if addTarget:

            # Adding the rectangle outlines
            rectTheta, rectPhi = zip(*addTarget, addTarget[0])  # Closing the rectangle
            plt.plot(rectTheta, rectPhi, color='red', linewidth=2, label='target pattern outline')
            plt.legend()

            if showDelta:
                # Add red text to the top right of the red rectangle
                textX = max(rectTheta)
                textY = max(rectPhi)
                plt.text(textX, textY, f'delta = {showDelta:.3f}', color='red', ha='left', va='bottom')
        plt.show()

    def plot4Excitation(self, excitation, addTarget=False, showDelta=None):
        """
        plot a far field for an arbitrary excitation
        :param excitation: the excitation (voltages on antennas)
        :param addTarget: target pattern limits
        :param showDelta: delta (synthesis error value)
        """
        synthField = self.fieldFromExcit(excitation)
        eThetaMax__Theta, eThetaMax__Phi, ePhiMax__Theta, ePhiMax__Phi = self.plotPreprocessing(synthField)
        self.plotBase(synthField[0], eThetaMax__Theta, eThetaMax__Phi, 'theta component of the radiated field',
                      addTarget, showDelta)
        self.plotBase(synthField[1], ePhiMax__Theta, ePhiMax__Phi, 'phi component of the radiated field', addTarget,
                      showDelta)

    def plotOptimal(self, goal):
        """
        plot the field following from the optimal excitation calculation
        :param goal: goal pattern index
        """
        target = [[self.TP.thetaLims[goal][0], self.TP.phiLims[goal][0]],
                  [self.TP.thetaLims[goal][0], self.TP.phiLims[goal][1]],
                  [self.TP.thetaLims[goal][1], self.TP.phiLims[goal][1]],
                  [self.TP.thetaLims[goal][1], self.TP.phiLims[goal][0]]]

        self.plot4Excitation(self.optimalExcitation[goal], addTarget=target,
                             showDelta=self.currentDeltas[goal])

    def plotGoal(self, goal):
        """
        plot a goal function
        :param goal: goal pattern index
        """
        thetaCenter = np.mean(self.TP.thetaLims[goal, :])
        phiCenter = np.mean(self.TP.phiLims[goal, :])
        goalField = [self.farFields[f'goal{goal + 1}__theta'], self.farFields[f'goal{goal + 1}__phi']]
        self.plotBase(goalField[0], thetaCenter, phiCenter, 'theta component of the goal pattern')
        self.plotBase(goalField[1], thetaCenter, phiCenter, 'phi component of the goal pattern')


def complexVecDF(phi1r, phi1i, theta1r, theta1i):
    """
    Generate complex vector from scalar real components
    :param phi1r:
    :param phi1i:
    :param theta1r:
    :param theta1i:
    :return:
    """
    return np.array([complex(theta1r, theta1i), complex(phi1r, phi1i)])
