from src.antennaArrays.AntennaArray import *
import datetime
class ArraySynthesis(AntennaArray):
    """
    Subclass of AntennaArray responsible for the synthesis algorithm and HFSS interfacing.
    """
    def __init__(self, TP: TargetPattern, OP: OptimizationParams, BD: BasicData, app = None):
        """
        :param TP, OP, BD: target, optimization and basic data objects.
        :param app: hfss handle input for enabling test mocking
        """
        super().__init__(TP, OP, BD)
        # HFSS  handle
        self.app = app
        self.fastGenerate(app)

    def fastGenerate(self, app = None) -> None:
        """
        Launch HFSS, generate the far fields dataframe containing the fields corresponding to unit excitations and the
        target fields and prepare for synthesis run.

        :param app: hfss handle input for enabling test mocking
        """
        if app is not None:
            self.app = app
        else:
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

    def generateFarFieldDF(self) -> None:
        """
        Set up the dataframe containing all far field data.
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

    def intCoeff(self) -> None:
        """
        Add the integration coefficients to the main dataframe.
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

    def hfssSetZeroExcit(self) -> None:
        """
        Set zero values for the excitations. Also serves as a check to see whether they are defined in the model.
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
        if np.abs(thStep - phStep) / thStep >= 0.01:
            raise ValueError("step size in theta and phi should be the same")

    def setAnalyze(self, geomParams: np.ndarray) -> None:
        """
        Set geometry and solve setup in HFSS.

        :param geomParams: geometric parameters
        """
        self.app.load_project(self.BD.designPath)
        for ind, param in enumerate(geomParams):
            self.app[self.OP.geomParamNames[ind]] = f'{param}mm'
        self.app.analyze()

    def readFields(self) -> None:
        """
        Read the fields after solution to dataframe 'self.farFields'.
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

    def cleanup(self) -> None:
        """
        Cleanup routine; close project and delete results.
        """
        self.app.close_project()
        # delete previous calc results to prevent save time snowballing
        projectFolder = (self.BD.designPath.split('.'))[0]
        projectFolder = projectFolder + '.aedtresults'
        if os.path.exists(projectFolder) and os.path.isdir(projectFolder):
            shutil.rmtree(projectFolder)

    def recalc(self, geomParams: np.ndarray) -> None:
        """
        Change input parameters of HFSS model, run the simulation and write the results to self.farFields.
        """
        self.setAnalyze(geomParams)
        self.readFields()
        self.cleanup()

    def simulateAndFetch(self, geomParams: np.ndarray) -> None:
        """
        Feed new geometry to HFSS model, run simulation and extract data.
        """
        # timer
        t1 = time.perf_counter()

        # call the API to calculate the far fields
        self.recalc(geomParams)
        self.simTime = time.perf_counter() - t1

    def writeLog(self, geomParams: np.ndarray) -> None:
        """
        Write iteration results into text files and append to the class' tracking lists
        :param geomParams: geometric parameters
        """


        dataFolderExists = os.path.exists(f'{self.BD.projectDir}/data')
        if not dataFolderExists:
            os.makedirs(f'{self.BD.projectDir}/data')
            print("Data folder created!")


        with open(f'{self.BD.projectDir}/data/deltas' + str(self.BD.processNum) + '.txt', 'a+') as f:
            f.write(str(self.currentDeltas) + '        ; ' + 'simulation runtime = ' + str(self.simTime) + '\n')
            f.write('\n')
        with open(f'{self.BD.projectDir}/data/positions' + str(self.BD.processNum) + '.txt', 'a+') as f:
            f.write(str(geomParams) + '\n')
            f.write('\n')
        with open(f'{self.BD.projectDir}/data/optimalExcitation' + str(self.BD.processNum) + '.txt', 'a+') as f:
            f.write(str(self.optimalExcitation) + '\n')
            f.write('\n')

        self.deltas.append(copy.deepcopy(self.currentDeltas))
        self.excitations.append(copy.deepcopy(self.optimalExcitation))
        self.positions.append(copy.deepcopy(geomParams))
        self.simTimes.append(copy.deepcopy(self.simTime))

    def synthStep(self, geomParams: np.ndarray, gradient: None) -> float:
        """
        Synthesis step, perform optimization for selected parameters on the input.
        :param geomParams: geometric parameters
        """
        self.simulateAndFetch(geomParams)
        self.findOptimalExcitation()
        self.writeLog(geomParams)
        return self.currentDeltas[-1]

    def pickleSaver(self, archiveName: str) -> None:
        """
        Save important fields to a pickle archive.

        :param archiveName: name-code for the archive
        """
        fieldsToSave = {'deltas': self.deltas, 'excitations': self.excitations, 'positions': self.positions,
                        'simTimes': self.simTimes, 'allX': self.allX, 'x0': self.OP.x0, 'antNorm': self.antNorm,
                        'farFields': self.farFields, 'TP': self.TP, 'BD': self.BD, 'OP': self.OP}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save the selected fields to a pickle archive
        with open(f'{self.BD.projectDir}/data/res{archiveName}{self.BD.processNum}_{timestamp}.pkl', 'wb') as file:
            pickle.dump(fieldsToSave, file)

    def simultaneousOptimizer(self) -> None:
        """
        Simultaneous optimizer definition.
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

    def successiveOptimizer(self, outerIters: int) -> None:
        """
        Successive optimizer definition

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
