import copy
import pickle
import shutil
import time

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import pandas as pd
from pyaedt import *

# degree to radian conversion
RAD2DEG = 1 / 180 * np.pi

IMPEDANCE_OF_FREE_SPACE = 376.73


def findProjectBase():
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up the directory tree
    while True:
        # Check if the venv directory exists in the current directory
        if os.path.exists(os.path.join(current_dir, 'venv')):
            return current_dir

        # Move up one level in the directory tree
        parent_dir = os.path.dirname(current_dir)

        # If we've reached the root of the file system without finding venv, stop
        if parent_dir == current_dir:
            #raise Exception("Project base directory with 'venv' not found.")
            return parent_dir

        current_dir = parent_dir



class TargetPattern:
    """
    Class describing the patterns to synthesize
    """

    def __init__(self, thetaCenters: List[float], phiCenters: List[float], thetaWidth: float, phiWidth: float,
                 polarization: str, angleMargin: float = 0.1, customPolarization: Optional[List[List[complex]]] = None):
        """
        Initialize the TargetPattern with center points, widths, and polarization details.

        :param thetaCenters: Lobe centers in theta direction (in degrees).
        :param phiCenters: Lobe centers in phi direction (in degrees).
        :param thetaWidth: Lobe width in theta direction (in degrees).
        :param phiWidth: Lobe width in phi direction (in degrees).
        :param polarization: Polarization type ('theta', 'phi', or 'both').
        :param angleMargin: Numerical padding for angular condition checks.
        :param customPolarization: List of lists, each containing two complex numbers, used for 'custom' polarization.
        """
        self.thetaCenters = thetaCenters
        self.phiCenters = phiCenters
        self.thetaWidth = thetaWidth
        self.phiWidth = phiWidth
        self.polarization = polarization
        self.angleMargin = angleMargin
        self.customPolarization = customPolarization
        # Calculate lobe boundaries
        self.lims = None
        self.dirVec = None

        self.calculateLobeBoundaries()

        # Calculate direction vectors based on polarization
        self.calculateDirectionVectors()

        # Number of goal patterns / directions
        self.goalNum = self.lims.shape[0]
        self.thetaLims = self.lims[:, 0]
        self.phiLims = self.lims[:, 1]

    def calculateLobeBoundaries(self) -> None:
        """
        Specify the angle interval characterizing the lobe.
        """
        lims = []
        for theta in self.thetaCenters:
            for phi in self.phiCenters:
                lims.append(self.calculateSingleLobe(theta, phi))
        if self.polarization == 'both':
            self.lims = np.array(lims + lims)
        else:
            self.lims = np.array(lims)

    def calculateSingleLobe(self, theta: float, phi: float) -> Tuple[List[float], List[float]]:
        """
        Get a single lobe interval from center point and lobe width.

        :param theta: Theta lobe center.
        :param phi: Phi lobe center.
        :return: Lobe interval as a tuple of two lists, each containing two floats.
        """
        return (
            [theta - self.thetaWidth / 2, theta + self.thetaWidth / 2],
            [phi - self.phiWidth / 2, phi + self.phiWidth / 2]
        )

    def calculateDirectionVectors(self) -> None:
        """
        Calculate array of direction vectors based on polarization requirements.
        """
        if self.polarization == 'theta':
            self.dirVec = np.array([[1, 0]] * len(self.lims))
        elif self.polarization == 'phi':
            self.dirVec = np.array([[0, 1]] * len(self.lims))
        elif self.polarization == 'both':
            self.dirVec = np.array([[1, 0]] * (len(self.lims) // 2) + [[0, 1]] * (len(self.lims) // 2))
        elif self.polarization == 'custom':
            if not self.customPolarization or len(self.customPolarization) != len(self.lims):
                raise ValueError(
                    "Custom polarization requires a list of lists (each with two complex numbers) with the same length as 'lims'.")
            self.dirVec = np.array(self.customPolarization)
        else:
            raise ValueError("Polarization should be 'theta', 'phi', 'both', or 'custom'")


class OptimizationParams:
    """
    Parameters for the optimizer.
    """

    def __init__(self, N: int, runTime: float, xMin: np.ndarray, xMax: np.ndarray, x0: np.ndarray,
                 geomParamNames: List[str], ampSymbol: str = 'c', phaseSymbol: str = 'p',
                 method=nlopt.GN_DIRECT_L_RAND, dualPort: bool = False):
        """
        :param N: number of antennas
        :param runTime: optimization time limit in seconds
        :param xMin: array of minimum limit for the optimization parameters
        :param xMax: -||- maximum
        :param x0: initial step of the optimization (may get overwritten in the successive algorithm)
        :param geomParamNames: names of the optimizable geometric params in the hfss model
        :param ampSymbol: symbol for the amplitudes in the hffs model
        :param phaseSymbol: symbol for the phases in the hffs model
        :param method: optimization method
        :param dualPort: whether there's two ports on the antenna
        """
        self.N = N
        self.dualPort = dualPort
        self.runTime = runTime

        if not (len(xMin) == len(xMax) == len(x0) == len(geomParamNames)):
            raise ValueError("Lengths of xMin, xMax, x0, and geomParamNames must be equal.")
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
        Basic input data for the synthesis.
    """

    def __init__(self, angStep: float, fileName: str, processNum: int = 1, maxPasses: Optional[int] = None):
        """
        :param angStep: angular resolution of the output far fields
        :param fileName: obvious
        :param processNum: process number (relevant for parallel runs), ignore for now
        :param maxPasses: maximum number of passes in the adaptive HFSS solver
        """
        # Anglular resolution of output
        self.angStep = angStep
        # Impedance of free space
        self.Z0 = IMPEDANCE_OF_FREE_SPACE
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
        # base project directory
        self.projectDir = findProjectBase()

class AntennaArray:
    """
        Object realizing the synthesis.
    """

    def __init__(self, TP: TargetPattern, OP: OptimizationParams, BD: BasicData):
        """
        :param TP, OP, BD: target, optimization and basic data objects.
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

        if self.OP.dualPort:
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

        # far field dataframe
        self.farFields = None

    def writeTargetPatterns(self) -> None:
        """
        Calculate and save all goal patterns to the main dataframe.
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

    def normalizeTargetPatterns(self) -> None:
        """
        Self-explanatory.
        """
        self.integrateTarget()
        for goal in range(1, self.TP.goalNum + 1):
            for component in self.BD.components:
                self.farFields[f'goal{goal}{component}'] = self.farFields[f'goal{goal}{component}'] / np.sqrt(
                    self.targetNorm[goal - 1])

    def innerProduct(self, var1: pd.Series, var2: pd.Series) -> float:
        """
        Inner product operator definition.

        :param var1: Operand, dataframe column.
        :param var2: Same as var1.
        :return: Computed inner product.
        """
        return self.BD.angStep ** 2 * RAD2DEG ** 2 * np.sum(
            self.farFields['intCoeff'] * np.sin(self.farFields.index.get_level_values('theta') * RAD2DEG) *
            np.conjugate(var2) * var1)

    def integrateAntFields(self) -> None:
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

    def normalizeAntFields(self) -> None:
        """
        Self-explanatory.
        """
        for ant in range(1, self.OP.N):
            for port in self.ports:
                for component in self.BD.components:
                    self.farFields[f'c{port}{ant}{component}'] = \
                        self.farFields[f'c{port}{ant}{component}'] / np.sqrt(
                            np.real(self.antNorm)[ant + self.portKey[port] - 1])

    def integrateMutual(self) -> None:
        """
        Function to calculate inner product between the fields corresponding to different excitations.
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

    def integrateMixed(self) -> None:
        """
        Function to calculate inner product between the fields corresponding to an excitations and the target pattern.
        """
        self.mixedTerm = np.zeros_like(self.mixedTerm)

        for goal in range(1, self.TP.goalNum + 1):
            for ant in range(1, self.OP.N + 1):
                for port in self.ports:
                    for component in self.BD.components:
                        self.mixedTerm[goal - 1, ant - 1 + self.portKey[port]] += \
                            self.innerProduct(self.farFields[f'goal{goal}{component}'],
                                              self.farFields[f'c{port}{ant}{component}'])

    def integrateTarget(self) -> None:
        """
        Function to calculate the 'norm' of the target pattern.
        """
        self.targetNorm = np.zeros_like(self.targetNorm)

        for goal in range(1, self.TP.goalNum + 1):
            for component in self.BD.components:
                self.targetNorm[goal - 1] += self.innerProduct(self.farFields[f'goal{goal}{component}'],
                                                               self.farFields[f'goal{goal}{component}'])

    def fieldFromExcit(self, excitation: Union[List[float], np.ndarray]) -> List[pd.Series]:
        """
        Calculate the radiated far field [theta, phi] components from the excitation vector.

        :param excitation: Vector of excitation voltages.
        :return: List containing radiated far field theta and phi components.
        """
        synthField = [0 * self.farFields['c1__theta'], 0 * self.farFields['c1__phi']]
        for ant in range(1, self.OP.N + 1):
            for port in self.ports:
                for ind, component in enumerate(self.BD.components):
                    synthField[ind] += excitation[ant - 1 + self.portKey[port]] * self.farFields[
                        f'c{port}{ant}{component}']
        return synthField

    def integrateByDef(self, excitations: List[Union[List[float], np.ndarray]]) -> np.ndarray:
        """
        Function to calculate the norm of the difference between a synthesized field weighed by the excitation
        coefficients and the target pattern.

        :param excitations: List of vectors, each containing the excitations of the individual antennas for a particular goal.
        :return: Norm of the difference as a NumPy array.
        """
        delt = np.zeros(self.TP.goalNum + 1)
        for goal in range(1, self.TP.goalNum + 1):
            synthField = self.fieldFromExcit(excitations[goal - 1])
            for ind, component in enumerate(self.BD.components):
                delt[goal - 1] += self.innerProduct(synthField[ind] - self.farFields[f'goal{goal}{component}'],
                                                    synthField[ind] - self.farFields[f'goal{goal}{component}'])
        delt[-1] = np.linalg.norm(delt[0: -1]) / np.sqrt(self.TP.goalNum)
        return delt

    def findOptimalExcitation(self) -> None:
        """
        Find the optimal excitation to achieve the desired goal patterns, refer to the formulas in the paper by Marak et al.
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

        # for debugging
        d0 = self.integrateByDef(self.optimalExcitation)

    def angleGrid(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get theta and phi multiindex values from a far field dataframe.

        :return: Grid points for theta and phi coordinates as a tuple of Pandas Series.
        """
        # Extract levels for phi and theta coordinates
        phiVals = self.farFields.index.get_level_values('phi')
        thetaVals = self.farFields.index.get_level_values('theta')
        return thetaVals, phiVals



