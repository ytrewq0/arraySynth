import os
import sys
from unittest.mock import patch
import unittest
import pandas as pd
import numpy as np

from numpy.random import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.antennaArrays.AntennaArray import AntennaArray, BasicData, TargetPattern, OptimizationParams
from src.antennaArrays.ArraySynthesis import ArraySynthesis
import src.antennaArrays


class FarFieldData:
    def __init__(self, picklePath):
        # Read the DataFrame from the pickle file
        self.full_matrix_real_imag = pd.read_pickle(picklePath)['farFields'][['c1__theta', 'c1__phi']]
        self.full_matrix_real_imag.rename(columns={'c1__theta': 'rETheta', 'c1__phi': 'rEPhi'}, inplace=True)
        self.full_matrix_real_imag = (self.full_matrix_real_imag.map(lambda x: np.real(x)),
                                      1j * self.full_matrix_real_imag.map(lambda x: np.imag(x)))

class MockApp(unittest.mock.MagicMock):
    def __init__(self, *args, picklePath = None, **kwargs):
        super().__init__(*args, **kwargs)
        if picklePath:
            self.picklePath = picklePath
            self.archive = pd.read_pickle(picklePath)
            array = AntennaArray(self.archive['TP'], self.archive['OP'], self.archive['BD'])
            self.excitationInputs = [f'{array.OP.ampSymbol}{port}{ind}' for port in array.ports for ind in range(1, array.OP.N + 1)]
            self.geometryInputs = array.OP.geomParamNames
            self.analysisRequired = False
            self.projectLoaded = False
            self.post = self.Post(self)
            self.field_setups = [self.FieldSetup(self)]

    def __setitem__(self, key, value):
        if key not in (self.excitationInputs + self.geometryInputs):
            raise ValueError(f"Invalid input: {key}")
        if key in self.geometryInputs:
            self.analysis_required = True

    def __getitem__(self, key):
        if key not in (self.excitationInputs +self.geometryInputs):
            raise ValueError(f"Input {key} not set")
        return random

    def analyze(self):
        if not self.projectLoaded:
            raise ValueError("no project is loaded")
        self.analysis_required = False

    def close_project(self):
        self.projectLoaded = False

    def load_project(self, path):
        self.projectLoaded = True

    class FieldSetup:
        def __init__(self, app):
            self.app = app
            self.phi_step = 1
            self.theta_step = 1

    class Post:
        def __init__(self, app):
            self.app = app

        def get_far_field_data(self, dataTypes):
            if set(dataTypes) != set(["rEPhi", "rETheta"]):
                raise ValueError("Invalid data types requested. Only ['rEPhi', 'rETheta'] is allowed.")
            # The method now properly checks for the specific arguments
            return FarFieldData(self.app.picklePath)

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create an instance of MockApp
        picklePath = "resSimult1_20240105_003823.pkl"  # Replace with the actual path
        cls.mockApp = MockApp(picklePath = picklePath)

        # Create an instance of ArraySynthesis with MockApp
        cls.synth = ArraySynthesis(cls.mockApp.archive['TP'], cls.mockApp.archive['OP'], cls.mockApp.archive['BD'],
                                   app = cls.mockApp)


    def testInit(self):
        self.synth.fastGenerate(app = self.mockApp)
        self.synth.setAnalyze(self.synth.OP.x0)

        a0 = self.synth.app[self.synth.OP.geomParamNames[0]]
        self.synth.app[self.synth.OP.geomParamNames[0]] = '0.1mm'
        a1 = self.synth.app[self.mockApp.excitationInputs[0]]
        self.synth.app[self.mockApp.excitationInputs[0]] = '0.4'


if __name__ == "__main__":
    unittest.main()