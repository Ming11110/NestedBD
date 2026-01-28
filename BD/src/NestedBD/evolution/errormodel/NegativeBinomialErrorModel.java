package NestedBD.evolution.errormodel;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.datatype.DataType;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import org.apache.commons.math.special.Gamma;

import static cern.jet.math.Arithmetic.logFactorial;

/**
 * Negative Binomial error model for integer copy-number data.
 * <p>
 * The model assumes:
 *  <ul>
 *      <li> If the true copy number is 0, the observed copy number is 0 with probability 1.</li>
 *      <li> If the true copy number is a > 0, the observed value X follows a Negative Binomial:</li>
 *         X ~ NB(mean = a, variance = a + a²/k)
 * </ul>
 * using the parameterisation:
 * <ul>
 *     <li> r = k </li>
 *     <li> p = k / (k + a) </li>
 * <p>
 * </ul>
 * with probability mass function:
 * <p>
 *     P(X = x) = Γ(x+r) / (x! Γ(r)) · (1-p)^x · p^r .
 * <p>
 * A Poisson approximation is used if k is large.
 */
@Description("Negative Binomial error model for copy-number data.")
public class NegativeBinomialErrorModel extends ErrorModel {

    public final Input<RealParameter> dispersionInput =
            new Input<>("dispersion", "Negative binomial dispersion parameter k.", Input.Validate.REQUIRED);

    public final Input<IntegerParameter> nstateInput =
            new Input<>("nstate", "Number of states (same as in BD model).", Input.Validate.REQUIRED);

    protected int nrOfStates;
    private RealParameter kParam;

    protected DataType dataType;

    @Override
    public void initAndValidate() {

        super.initAndValidate();

        // Get parameters
        kParam = dispersionInput.get();

        if (kParam.getValue() <= 0) {
            throw new IllegalArgumentException("dispersion k must be > 0.");
        }

        // Number of CN states
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
            double colSum = 0.0;

            // For each observed CN (rows)
            for (int obs = 0; obs < nrOfStates; obs++) {
                double prob;
                if (trueState == 0) {
                    prob = (obs == 0) ? 1.0 : 0.0;
                } else {
                    prob = negativeBinomialPMF(obs, trueState);
                }

                errorMatrix[obs][trueState] = prob;
                colSum += prob;
            }

            // Normalize column to sum to 1
            if (colSum > 0.0) {
                for (int obs = 0; obs < nrOfStates; obs++) {
                    errorMatrix[obs][trueState] /= colSum;
                }
            } else {
                errorMatrix[trueState][trueState] = 1.0;
            }
        }
    }


    private double negativeBinomialPMF(int observedCN, int trueCN) {

        if (trueCN <= 0) return 0.0;  // handled in zero-inflation block
        if (observedCN < 0) return 0.0;

        double mean = (double) trueCN;
        double k = kParam.getValue();

        // Poisson limit for large k
        if (k > 1e6) {
            double logProb = observedCN * Math.log(mean)
                    - mean
                    - logFactorial(observedCN);
            return Math.exp(logProb);
        }

        // NB parameters
        double r = k;
        double p = k / (k + mean);
        int x = observedCN;

        try {
            // log P(x)
            double logProb =
                    Gamma.logGamma(x + r)
                            - logFactorial(x)
                            - Gamma.logGamma(r)
                            + x * Math.log(1 - p)
                            + r * Math.log(p);

            return Math.exp(logProb);

        } catch (Exception e) {
            return 1e-12;
        }
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

        // Return pre-computed value
        return errorMatrix[observedState][trueState];
    }

    @Override
    public double[] getProbabilities(int observedState) {
        // Update matrix if needed
        ensureMatrixUpdated();

        double[] p = new double[nrOfStates];

        // Copy from pre-computed matrix
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
        if (datatype instanceof beast.base.evolution.datatype.IntegerData) {
            return true;
        }
        return datatype != null && datatype.getStateCount() > 0;
    }

    @Override
    public boolean requiresRecalculation() {
        updateMatrix = true;
        return true;
    }
}
