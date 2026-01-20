package NestedBD.evolution.errormodel;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.datatype.DataType;
import beast.base.inference.parameter.RealParameter;

/**
 * Discrete Gaussian error model for integer copy-number data.
 * <p>
 * The model assumes:
 *  <ul>
 *      <li> If the true copy number is 0, the observed copy number is 0 with probability 1.</li>
 *      <li> If the true copy number is μ > 0, the observed value X follows: </li>
 *         P(X = x | μ, σ) = exp(-(x-μ)²/(2σ²)) / Z(μ, σ)
 *      <li> where Z(μ, σ) = Σ(i=0 to nstate-1) exp(-(i-μ)²/(2σ²)) is the normalization constant </li>
 * </ul>
 * σ is the measurement error standard deviation.
 * <ul>
 *     <li>When σ → 0, all probability mass concentrates at the true value (no error).</li>
 *     <li>When σ is large, the distribution is wide (high measurement noise).</li>
 * </ul>
 * </p>
 */
@Description("Discrete Gaussian (Truncated Normal) error model for copy-number data.")
public class DiscreteGaussianErrorModel extends ErrorModel {

    public final Input<RealParameter> sigmaInput =
            new Input<>("sigma",
                    "Gaussian error standard deviation (σ). When σ=0, observations equal true values.",
                    Input.Validate.REQUIRED);

    public final Input<RealParameter> nstateInput =
            new Input<>("nstate",
                    "Number of copy number states (must be >= max observed CN + 1).",
                    Input.Validate.REQUIRED);

    protected int nrOfStates;
    private RealParameter sigmaParam;
    protected DataType dataType;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        // Get sigma parameter
        sigmaParam = sigmaInput.get();
        if (sigmaParam.getValue() < 0) {
            throw new IllegalArgumentException("sigma must be >= 0.");
        }

        // Get number of CN states
        if (nstateInput.get() != null) {
            nrOfStates = (int) Math.round(nstateInput.get().getValue());
        } else if (dataType != null) {
            nrOfStates = dataType.getStateCount();
        } else {
            throw new IllegalArgumentException("nstate or datatype is required.");
        }

        if (nrOfStates <= 1) {
            throw new IllegalArgumentException("nrOfStates must be >= 2");
        }

        // Initialize error matrix
        setupErrorMatrix();
        updateMatrix = false;
    }

    @Override
    public void setupErrorMatrix() {
        if (errorMatrix == null) {
            errorMatrix = new double[nrOfStates][nrOfStates];
        }

        // For each true CN (columns)
        for (int trueState = 0; trueState < nrOfStates; trueState++) {

            // Special case: true CN = 0 (deletion is deterministic)
            if (trueState == 0) {
                for (int obs = 0; obs < nrOfStates; obs++) {
                    errorMatrix[obs][0] = (obs == 0) ? 1.0 : 0.0;
                }
                continue;
            }

            // For true CN > 0: compute discrete Gaussian probabilities
            double[] probs = computeDiscreteGaussian(trueState);

            for (int obs = 0; obs < nrOfStates; obs++) {
                errorMatrix[obs][trueState] = probs[obs];
            }
        }
    }

    /**
     * <p>
     * <ul> Compute discrete Gaussian probabilities for a given true state μ.
     * <li>Returns P(X = x | μ, σ) for x = 0, 1, ..., nrOfStates-1 </li> </ul>
     * <ul> Formula: P(x | μ, σ) = exp(-(x-μ)²/(2σ²)) / Z(μ, σ)
     * <li> where Z(μ, σ) is the normalization constant </li></ul>
     * </p>
     */
    private double[] computeDiscreteGaussian(int mu) {
        double sigma = sigmaParam.getValue();
        double[] probs = new double[nrOfStates];

        // Special case: if sigma very small, put all mass at true value
        if (sigma < 1e-6) {
            if (mu < nrOfStates) {
                probs[mu] = 1.0;
            }
            return probs;
        }

        // Compute unnormalized weights: exp(-(x-μ)²/(2σ²))
        double sum = 0.0;
        for (int x = 0; x < nrOfStates; x++) {
            double diff = x - mu;
            probs[x] = Math.exp(-(diff * diff) / (2.0 * sigma * sigma));
            sum += probs[x];
        }

        // Normalize to make probabilities sum to 1
        if (sum > 0) {
            for (int x = 0; x < nrOfStates; x++) {
                probs[x] /= sum;
            }
        } else {
            // Fallback: if all probabilities are essentially 0 (shouldn't happen)
            if (mu < nrOfStates) {
                probs[mu] = 1.0;
            }
        }

        return probs;
    }

    /**
     * Ensure error matrix is up-to-date before use
     */
    private void ensureMatrixUpdated() {
        if (updateMatrix) {
            setupErrorMatrix();
            updateMatrix = false;
        }
    }

    @Override
    public double getProbability(int observedState, int trueState) {
        // Update matrix if needed
        ensureMatrixUpdated();

        // Validate inputs
        if (observedState < 0 || observedState >= nrOfStates) {
            return 0.0;
        }
        if (trueState < 0 || trueState >= nrOfStates) {
            return 0.0;
        }

        return errorMatrix[observedState][trueState];
    }

    @Override
    public double[] getProbabilities(int observedState) {
        // Update matrix if neededz
        ensureMatrixUpdated();

        double[] p = new double[nrOfStates];

        // Copy the entire row from the pre-computed matrix
        for (int trueState = 0; trueState < nrOfStates; trueState++) {
            p[trueState] = errorMatrix[observedState][trueState];
        }
        return p;
    }

    @Override
    public double[] getProbabilities(int observedState, double w, int totalread) {
        // For discrete Gaussian, we don't use w and totalread
        return getProbabilities(observedState);
    }

    @Override
    public boolean canHandleDataType(DataType datatype) {
        // Can handle IntegerData or any datatype with defined state count
        if (datatype instanceof beast.base.evolution.datatype.IntegerData) {
            return true;
        }
        return datatype != null && datatype.getStateCount() > 0;
    }

    @Override
    public boolean requiresRecalculation() {
        // Mark that error matrix needs recalculation
        // This is called when parameters (sigma) change during MCMC
        updateMatrix = true;
        return true;
    }

    // Getters for parameters (useful for logging/debugging)
    public double getSigma() {
        return sigmaParam.getValue();
    }

    public int getNrOfStates() {
        return nrOfStates;
    }
}