package phylonco.lphy.spi;

import lphy.base.spi.LPhyBaseImpl;
import lphy.core.model.BasicFunction;
import lphy.core.model.GenerativeDistribution;
//import phylonco.lphy.evolution.alignment.*;
import phylonco.lphy.evolution.copynumbermodel.*;

import java.util.Arrays;
import java.util.List;

/**
 * The provider of SPI which is an implementation of a service.
 * It requires a public no-args constructor.
 *
 * @author Walter Xie
 */
public class NestedBDImpl extends LPhyBaseImpl {

    /**
     * Required by ServiceLoader.
     */
    public NestedBDImpl() {
        //TODO print package or classes info here?
    }

    @Override
    public List<Class<? extends GenerativeDistribution>> declareDistributions() {
        return Arrays.asList(
                PhyloDiscrete.class,
                NegativeBinomialErrorModel.class,
                DiscreteGaussianErrorModel.class
        );
    }

    @Override
    public List<Class<? extends BasicFunction>> declareFunctions() {
        return Arrays.asList(
                CopyNumberBD.class,
                ReadCopyProfile.class
        );
    }

    public String getExtensionName() {
        return "NestedBD lphy library";
    }
}
