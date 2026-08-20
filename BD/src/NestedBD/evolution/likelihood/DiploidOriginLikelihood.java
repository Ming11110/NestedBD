package NestedBD.evolution.likelihood;

import beast.base.core.Input;
import beast.base.evolution.likelihood.BeerLikelihoodCore;
import beast.base.evolution.likelihood.BeerLikelihoodCore4;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.core.Input.Validate;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;

import java.util.*;

public class DiploidOriginLikelihood extends TreeLikelihood {
    public Input<RealParameter> origtime = new Input<RealParameter>("origtime", "time between diploid and tree root");
    public Input<IntegerParameter> nstates = new Input<IntegerParameter>("nstates", "same as what in BD model", Validate.REQUIRED);
    //protected int nrofStates = dataInput.get().getMaxStateCount();
    //protected int nrofPattern = dataInput.get().getPatternCount();
    //protected int nrofStates = 30;
    protected int nrofStates;
    protected static double bd_rate = 0.01;
    protected boolean recalc = true;

    /* Override to avoid BEAGLE */
    @Override
    public void initAndValidate() {
        // sanity check: Make sure data is an Alignment
        if (!(dataInput.get() instanceof Alignment)) {
            throw new RuntimeException("Expected Alignment as data, not " + dataInput.get().getClass().getName());
        }
        alignment = (Alignment) dataInput.get();

        // sanity check: alignment should have same #taxa as tree
        if (alignment.getTaxonCount() != treeInput.get().getLeafNodeCount()) {
            throw new IllegalArgumentException("The number of nodes in the tree does not match the number of sequences");
        }

        /**
         * BEAGLE is disabled here because the current copy-number substitution model
         * does not provide a rate matrix (Q) or eigen decomposition
         **/
        beagle = null;
        Log.info.println("Skipping BEAGLE for IntegerData - using Java likelihood core");

        int nodeCount = treeInput.get().getNodeCount();
        if (!(siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        }

        // Initialize the site and substitution models
        m_siteModel = (SiteModel.Base) siteModelInput.get();
        m_siteModel.setDataType(alignment.getDataType());
        substitutionModel = m_siteModel.substModelInput.get();

        if (branchRateModelInput.get() != null) {
            branchRateModel = branchRateModelInput.get();
        } else {
            branchRateModel = new StrictClockModel();
        }

        m_branchLengths = new double[nodeCount];
        storedBranchLengths = new double[nodeCount];


        // sanity check: nstates
        if (nstates.get() == null) {
            throw new IllegalArgumentException("number of states to consider is required");
        }
        nrofStates = nstates.get().getValue();

        //Set up the likelihood computation core
        int stateCount = nrofStates;
        int patterns = alignment.getPatternCount();

        if (stateCount == 4) {
            likelihoodCore = new BeerLikelihoodCore4();
        } else {
            likelihoodCore = new BeerLikelihoodCore(stateCount);
        }

        // Report which likelihood core is being used
        String className = getClass().getSimpleName();
        Log.info.println(className + "(" + getID() + ") uses " + likelihoodCore.getClass().getSimpleName());
        Log.info.println("  " + alignment.toString(true));

        // Handle invariant sites
        proportionInvariant = m_siteModel.getProportionInvariant();
        m_siteModel.setPropInvariantIsCategory(false);
        if (proportionInvariant > 0) {
            calcConstantPatternIndices(patterns, stateCount);
        }

        initCore();

        patternLogLikelihoods = new double[patterns];
        m_fRootPartials = new double[patterns * stateCount];
        matrixSize = (stateCount + 1) * (stateCount + 1);
        probabilities = new double[(stateCount + 1) * (stateCount + 1)];
        Arrays.fill(probabilities, 1.0);

        if (alignment.isAscertained) {
            useAscertainedSitePatterns = true;
        }
    }

    @Override
    protected boolean requiresRecalculation() {
        //final TreeInterface tree = treeInput.get();
        //System.out.println(origtime.get().somethingIsDirty());
        //System.out.println(traverse(tree.getRoot()) != Tree.IS_CLEAN);
        return (origtime.get().somethingIsDirty() | recalc);

        //return true;
    }

    /**
     * Calculate the log likelihood of the current state.
     *
     * @return the log likelihood.
     */
    double m_fScale = 1.01;
    int m_nScale = 0;
    int X = 100;

    @Override
    public double calculateLogP() {
        if (beagle != null) {
            logP = beagle.calculateLogP();
            return logP;
        }
        final TreeInterface tree = treeInput.get();

        try {
        	if (traverse(tree.getRoot()) != Tree.IS_CLEAN)
        		calcLogP();
        }
        catch (ArithmeticException e) {
            System.out.println("exception occurred");
            return Double.NEGATIVE_INFINITY;
        }
        m_nScale++;
        if (logP > 0 || (likelihoodCore.getUseScaling() && m_nScale > X)) {
//            System.err.println("Switch off scaling");
//            m_likelihoodCore.setUseScaling(1.0);
//            m_likelihoodCore.unstore();
//            m_nHasDirt = Tree.IS_FILTHY;
//            X *= 2;
//            traverse(tree.getRoot());
//            calcLogP();
//            return logP;
        } else if (logP == Double.NEGATIVE_INFINITY && m_fScale < 10 && !scaling.get().equals(Scaling.none)) { // && !m_likelihoodCore.getUseScaling()) {
            m_nScale = 0;
            m_fScale *= 1.01;
            Log.warning.println("Turning on scaling to prevent numeric instability " + m_fScale);
            likelihoodCore.setUseScaling(m_fScale);
            likelihoodCore.unstore();
            hasDirt = Tree.IS_FILTHY;
            traverse(tree.getRoot());
            calcLogP();
            return logP;
        }
        return logP;
    }


    @Override
    public void calcLogP() {
        logP = 0.0;
        if (useAscertainedSitePatterns) {
            final double ascertainmentCorrection = dataInput.get().getAscertainmentCorrection(patternLogLikelihoods);
            for (int i = 0; i < dataInput.get().getPatternCount(); i++) {
                logP += (patternLogLikelihoods[i] - ascertainmentCorrection) * dataInput.get().getPatternWeight(i);
            }
        } else {
            for (int i = 0; i < dataInput.get().getPatternCount(); i++) {
                logP += patternLogLikelihoods[i] * dataInput.get().getPatternWeight(i);
                //System.out.println(patternLogLikelihoods[i]);
            }
        }
    }
    
    /**
     * set leaf partials in likelihood core *
     */

    // for testing

    
    /* Assumes there IS a branch rate model as opposed to traverse() 
     * compute the diploid origin likelihood instead of likelihood
     */
    @Override
    protected int traverse(final Node node) {
        int update = (node.isDirty() | hasDirt);
        final int nodeIndex = node.getNr();
        final double branchRate = branchRateModel.getRateForBranch(node);
        final double branchTime = node.getLength() * branchRate;


        // First update the transition probability matrix(ices) for this branch
        //if (!node.isRoot() && (update != Tree.IS_CLEAN || branchTime != m_StoredBranchLengths[nodeIndex])) {
        if (!node.isRoot() && (update != Tree.IS_CLEAN || branchTime != m_branchLengths[nodeIndex])) {
            m_branchLengths[nodeIndex] = branchTime;
            final Node parent = node.getParent();
            likelihoodCore.setNodeMatrixForUpdate(nodeIndex);
            for (int i = 0; i < m_siteModel.getCategoryCount(); i++) {
                final double jointBranchRate = m_siteModel.getRateForCategory(i, node) * branchRate;
                substitutionModel.getTransitionProbabilities(node, parent.getHeight(), node.getHeight(), jointBranchRate, probabilities);
                //if (node.getNr() == 29) {
                	//System.out.println(node.getNr() + " " + Arrays.toString(probabilities));
                //}
                
                likelihoodCore.setNodeMatrix(nodeIndex, i, probabilities);
            }
            update |= Tree.IS_DIRTY;
        }

        // If the node is internal, update the partial likelihoods.
        if (!node.isLeaf()) {
            // Traverse down the two child nodes
            final Node child1 = node.getLeft(); //Two children
            final int update1 = traverse(child1);

            final Node child2 = node.getRight();
            final int update2 = traverse(child2);

            // If either child node was updated then update this node too
            if (update1 != Tree.IS_CLEAN || update2 != Tree.IS_CLEAN) {

                final int childNum1 = child1.getNr();
                final int childNum2 = child2.getNr();

                likelihoodCore.setNodePartialsForUpdate(nodeIndex);
                update |= (update1 | update2);
                if (update >= Tree.IS_FILTHY) {
                    likelihoodCore.setNodeStatesForUpdate(nodeIndex);
                }
                
                

                if (m_siteModel.integrateAcrossCategories()) {
                	//System.out.println(nodeIndex);
                    likelihoodCore.calculatePartials(childNum1, childNum2, nodeIndex);
                } else {
                    throw new RuntimeException("Error TreeLikelihood 201: Site categories not supported");
                    //m_pLikelihoodCore->calculatePartials(childNum1, childNum2, nodeNum, siteCategories);
                }

                if (node.isRoot()) {
                	//System.out.println("root");
                    // No parent this is the root of the beast.tree -
                    // calculate the pattern likelihoods
                    final double[] proportions = m_siteModel.getCategoryProportions(node);
                    //System.out.println(node.getNr() + " Proportions" + Arrays.toString(proportions));
                    likelihoodCore.integratePartials(node.getNr(), proportions, m_fRootPartials);
                    //System.out.println(node.getNr() + " m_RootPartials " + Arrays.toString(m_fRootPartials));
                    if (getConstantPattern() != null) { // && !SiteModel.g_bUseOriginal) {
                        proportionInvariant = m_siteModel.getProportionInvariant();
                        // some portion of sites is invariant, so adjust root partials for this
                        for (final int i : getConstantPattern()) {
                            m_fRootPartials[i] += proportionInvariant;
                        }
                    }

                    double distance = origtime.get().getValue();
                    //System.out.println("DD" + distance);
                    substitutionModel.getTransitionProbabilities(node, distance, 0.0, 1.0, probabilities);
                    //System.out.println("probabilities" + Arrays.toString(probabilities));
                    DipOrigLikelihoods(m_fRootPartials, probabilities, patternLogLikelihoods);
                }

            }
        }
        if (update == 0) {
        	recalc = false;
        }
        return update;
    } // traverseWithBRM
    

    public void DipOrigLikelihoods(double[] partials, double[] transition_prob, double[] outLogLikelihoods) {
    	
        int v = 0;
        //System.out.println("Where");
        //double distance = origtime.get().getValue();
        //System.out.println("DDD" + distance);
        //System.out.println(dataInput.get().getPatternCount());
        for (int k = 0; k < dataInput.get().getPatternCount(); k++) {
            //double max_prob =  Double.NEGATIVE_INFINITY;
            double max_prob = 0;
            for (int i = 0; i < nrofStates; i++) {
                //System.out.println(DipOrigProb(i, distance));
                //System.out.println(transition_prob[62+i]);
                //max_prob = Double.max(max_prob, partials[v]* transition_prob[2 * nrofStates +i]);
                if (Double.isNaN(partials[v])) {
                    System.out.println(k);
                } else {
                    max_prob += partials[v] * transition_prob[2 * nrofStates + i];
                }
            	//if (k == dataInput.get().getPatternCount()-1) {
            		//System.out.println(partials[v]);}
                v++;
            }

            outLogLikelihoods[k] = Math.log(max_prob) + likelihoodCore.getLogScalingFactor(k);
        }
    }
}
