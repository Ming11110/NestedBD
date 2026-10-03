package nestedBD.lphybeast.tobeast.generators;

import nestedBD.evolution.likelihood.DiploidOriginLikelihoodWithError;
import nestedBD.evolution.substitutionmodel.BD;
import beast.base.core.BEASTInterface;
import beast.base.evolution.datatype.DataType;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.inference.parameter.IntScalarParam;
import beast.base.spec.domain.PositiveInt;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.NonNegativeReal;

import lphy.core.model.GraphicalModelNode;
import lphy.core.model.Value;
import lphybeast.BEASTContext;
import lphybeast.GeneratorToBEAST;
import lphybeast.tobeast.generators.PhyloCTMCToBEAST;
import nestedBD.lphy.evolution.copynumbermodel.*;
import nestedBD.evolution.errormodel.NegativeBinomialErrorModel;

import java.util.Objects;

public class NegativeBinomialErrorModelToBEAST
        implements GeneratorToBEAST<nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel,
        DiploidOriginLikelihoodWithError> {

    @Override
    public DiploidOriginLikelihoodWithError generatorToBEAST(
            nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel generator,
            BEASTInterface value,
            BEASTContext context) {

        // Step 1: Get the error alignment (observed data with error)
        assert value instanceof beast.base.evolution.alignment.Alignment;
        beast.base.evolution.alignment.Alignment errAlignment =
                (beast.base.evolution.alignment.Alignment) value;

        // Step 2: Find the PhyloDiscrete generator (to get tree and model info)
        PhyloDiscrete phyloDiscrete = findPhyloDiscrete(generator);

        // Step 3: Get nstate from the CopyNumberBD model
        int nstate = getNstateFromModel(phyloDiscrete);

        // Step 4: Remove the original likelihood objects
        removeOriginalLikelihood(generator, phyloDiscrete, context);

        // Step 5: Create the BEAST2 NegativeBinomialErrorModel
        NegativeBinomialErrorModel beastErrorModel = new NegativeBinomialErrorModel();

        DataType beastDataType = errAlignment.getDataType();
        beastErrorModel.setInputValue("datatype", beastDataType);

        RealScalarParam<PositiveReal> dispersionParam =
                (RealScalarParam<PositiveReal>) context.getAsRealScalar(generator.getDispersion());
        beastErrorModel.setInputValue("dispersion", dispersionParam);

        beastErrorModel.setInputValue("nstate", new IntScalarParam<>(nstate, PositiveInt.INSTANCE));
        beastErrorModel.initAndValidate();

        // Step 6: Create DiploidOriginLikelihoodWithError
        DiploidOriginLikelihoodWithError likelihoodWithError =
                createLikelihoodWithError(errAlignment, beastErrorModel, phyloDiscrete, nstate, context);

        // Step 7: Add to logging
        context.addExtraLoggable(likelihoodWithError);

        return likelihoodWithError;
    }

    /**
     * Find the PhyloDiscrete generator by tracing back through the alignment input
     */
    private PhyloDiscrete findPhyloDiscrete(
            nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel generator) {

        for (GraphicalModelNode<?> input : Objects.requireNonNull(generator.getInputs())) {
            if (input instanceof Value && input.value() instanceof IntegerCharacterMatrix) {
                Value<IntegerCharacterMatrix> alignmentValue = (Value<IntegerCharacterMatrix>) input;
                if (alignmentValue.getGenerator() instanceof PhyloDiscrete) {
                    return (PhyloDiscrete) alignmentValue.getGenerator();
                }
            }
        }
        throw new IllegalArgumentException("Cannot find PhyloDiscrete!");
    }

    /**
     * Extract nstate from the CopyNumberBD model
     */
    private int getNstateFromModel(PhyloDiscrete phyloDiscrete) {
        Value<MarkovTraitEvolution<Integer>> modelValue = phyloDiscrete.getModel();

        if (modelValue.value() instanceof CopyNumberBD copyNumberBD) {
            return copyNumberBD.getNstate().value();
        } else {
            throw new IllegalArgumentException("Expected CopyNumberBD model but found: " +
                    modelValue.value().getClass().getName());
        }
    }

    /**
     * Create and configure DiploidOriginLikelihoodWithError
     */
    private DiploidOriginLikelihoodWithError createLikelihoodWithError(
            beast.base.evolution.alignment.Alignment errAlignment,
            NegativeBinomialErrorModel errorModel,
            PhyloDiscrete phyloDiscrete,
            int nstate,
            BEASTContext context) {

        DiploidOriginLikelihoodWithError likelihoodWithError = new DiploidOriginLikelihoodWithError();

        // Set the error alignment (observed data)
        likelihoodWithError.setInputValue("data", errAlignment);

        // Set the error model
        likelihoodWithError.setInputValue("errorModel", errorModel);

        // Get and set the tree
        PhyloCTMCToBEAST.constructTreeAndBranchRate(phyloDiscrete, likelihoodWithError, context, false);

        // Set origin time
        PhyloDiscreteToBEAST.setOrigtime(phyloDiscrete, likelihoodWithError, context);

        // Set nstate
        likelihoodWithError.setInputValue("nstates", new IntScalarParam<>(nstate, PositiveInt.INSTANCE));

        // Get the BD model and create site model
        Value<MarkovTraitEvolution<Integer>> modelValue = phyloDiscrete.getModel();
        BD bdModel = (BD) context.getBEASTObject(modelValue);

        SiteModel siteModel = new SiteModel();
        siteModel.setInputValue("substModel", bdModel);
        siteModel.initAndValidate();
        likelihoodWithError.setInputValue("siteModel", siteModel);

        // Initialize
        likelihoodWithError.initAndValidate();

        return likelihoodWithError;
    }

    /**
     * Remove the original alignment and likelihood that are replaced by the error model version
     */
    private void removeOriginalLikelihood(
            nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel generator,
            PhyloDiscrete phyloDiscrete,
            BEASTContext context) {

        // Find and remove the original alignment
        for (GraphicalModelNode<?> input : Objects.requireNonNull(generator.getInputs())) {
            if (input instanceof Value && input.value() instanceof IntegerCharacterMatrix) {
                Value<IntegerCharacterMatrix> origAlignmentInput = (Value<IntegerCharacterMatrix>) input;
                BEASTInterface beastOrigAlignment = context.getBEASTObject(origAlignmentInput);
                context.removeBEASTObject(beastOrigAlignment);
                break;
            }
        }

        // Remove the original likelihood
        BEASTInterface originalLikelihood = context.getBEASTObject(phyloDiscrete);
        context.removeBEASTObject(originalLikelihood);
    }

    @Override
    public Class<nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel> getGeneratorClass() {
        return nestedBD.lphy.evolution.copynumbermodel.NegativeBinomialErrorModel.class;
    }

    @Override
    public Class<DiploidOriginLikelihoodWithError> getBEASTClass() {
        return DiploidOriginLikelihoodWithError.class;
    }
}