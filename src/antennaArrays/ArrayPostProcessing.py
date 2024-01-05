from src.antennaArrays.AntennaArray import *

class ArrayPostProcessing(AntennaArray):
    """
    Subclass of AntennaArray responsible for the synthesis algorithm and HFSS interfacing.
    """
    def __init__(self, archivePath: str):
        """
        :param archivePath: pickle archive to load
        """
        fields = pd.read_pickle(archivePath)

        super().__init__(fields['TP'], fields['OP'], fields['BD'])

        for key, value in fields.items():
            setattr(self, key, value)


    def maxFind4Plot(self, field: List[pd.Series], maxTheta: Optional[Union[float, int]] = None,
                     maxPhi: Optional[Union[float, int]] = None) -> Tuple[float, float, float, float]:
        """
        Find maximum values of the given field in terms of spherical angles theta and phi.

        :param field: A column of the far field dataframe.
        :param maxTheta: Manual input for the maximum, overrides the maximization routine.
        :param maxPhi: Analogous to maxTheta.
        :return: Tuple of maximum values for eThetaMax__Theta, eThetaMax__Phi, ePhiMax__Theta, ePhiMax__Phi.
        """
        thetaVals, phiVals = self.angleGrid()
        # Find the indices of the maximum value in the synthetic field
        maxIndTheta = np.argmax(np.abs(field[0]))
        maxIndPhi = np.argmax(np.abs(field[1]))

        if maxTheta is not None:
            eThetaMax__Theta = maxTheta
            ePhiMax__Theta = maxTheta
        else:
            eThetaMax__Theta = thetaVals[maxIndTheta]
            ePhiMax__Theta = thetaVals[maxIndPhi]

        if maxPhi is not None:
            eThetaMax__Phi = maxPhi
            ePhiMax__Phi = maxPhi
        else:
            eThetaMax__Phi = phiVals[maxIndTheta]
            ePhiMax__Phi = phiVals[maxIndPhi]

        return eThetaMax__Theta, eThetaMax__Phi, ePhiMax__Theta, ePhiMax__Phi

    def plotBase(self, fieldComponent: Union[pd.Series, List[float]], maxTheta: float, maxPhi: float,
                 title: str, addTarget: Union[bool, List[Tuple[float, float]]] = False,
                 showDelta: Union[bool, float] = False) -> None:
        """
        Plotting function.

        :param fieldComponent: Column of the far field dataframe.
        :param maxTheta: The coordinate of fieldComponent's maximum absolute value.
        :param maxPhi: Analogous to maxTheta.
        :param title: Plot title.
        :param addTarget: Target pattern limits or False if not used.
        :param showDelta: Delta (synthesis error value) or False if not used.
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

    def plot4Excitation(self, excitation: Union[np.ndarray, List[float]],
                        addTarget: Union[bool, List[List[float]]] = False,
                        showDelta: Union[bool, float] = False) -> None:
        """
        Plot a far field for an arbitrary excitation.

        :param excitation: The excitation (voltages on antennas).
        :param addTarget: Target pattern limits or False if not used.
        :param showDelta: Delta (synthesis error value) or None if not used.
        """
        synthField = self.fieldFromExcit(excitation)
        eThetaMax__Theta, eThetaMax__Phi, ePhiMax__Theta, ePhiMax__Phi = self.maxFind4Plot(synthField)
        self.plotBase(synthField[0], eThetaMax__Theta, eThetaMax__Phi, 'theta component of the radiated field',
                      addTarget, showDelta)
        self.plotBase(synthField[1], ePhiMax__Theta, ePhiMax__Phi, 'phi component of the radiated field', addTarget,
                      showDelta)

    def plotOptimal(self, goal: int) -> None:
        """
        Plot the field resulting from the optimal excitation calculation.

        :param goal: Goal pattern index.
        """
        target = [[self.TP.thetaLims[goal][0], self.TP.phiLims[goal][0]],
                  [self.TP.thetaLims[goal][0], self.TP.phiLims[goal][1]],
                  [self.TP.thetaLims[goal][1], self.TP.phiLims[goal][1]],
                  [self.TP.thetaLims[goal][1], self.TP.phiLims[goal][0]]]

        self.plot4Excitation(self.optimalExcitation[goal], addTarget=target,
                             showDelta=self.currentDeltas[goal])

    def plotGoal(self, goal: int) -> None:
        """
        Plot a goal function.

        :param goal: Goal pattern index.
        """
        thetaCenter = np.mean(self.TP.thetaLims[goal, :])
        phiCenter = np.mean(self.TP.phiLims[goal, :])
        goalField = [self.farFields[f'goal{goal + 1}__theta'], self.farFields[f'goal{goal + 1}__phi']]
        self.plotBase(goalField[0], thetaCenter, phiCenter, 'Theta component of the goal pattern')
        self.plotBase(goalField[1], thetaCenter, phiCenter, 'Phi component of the goal pattern')
