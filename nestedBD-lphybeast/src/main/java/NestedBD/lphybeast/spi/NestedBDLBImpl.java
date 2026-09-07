package nestedBD.lphybeast.spi;

import beast.base.evolution.datatype.DataType;
import jebl.evolution.sequences.SequenceType;
import lphy.core.model.Generator;
import lphybeast.GeneratorToBEAST;
import lphybeast.ValueToBEAST;
import lphybeast.spi.LPhyBEASTMapping;

import nestedBD.lphybeast.tobeast.generators.DiscreteGaussianErrorModelToBEAST;
import nestedBD.lphybeast.tobeast.generators.NegativeBinomialErrorModelToBEAST;
import nestedBD.lphybeast.tobeast.generators.PhyloDiscreteToBEAST;
import nestedBD.lphybeast.tobeast.values.CopyNumberBDToBEAST;
import nestedBD.lphy.evolution.copynumbermodel.CopyNumberBD;
import nestedBD.lphy.evolution.copynumbermodel.ReadCopyProfile;
import nestedBD.lphybeast.tobeast.values.IntegerCharacterMatrixToBEAST;

import java.util.List;
import java.util.Map;

public class NestedBDLBImpl implements LPhyBEASTMapping {
    @Override
    public List<Class<? extends GeneratorToBEAST>> getGeneratorToBEASTs() {
        return List.of(
               DiscreteGaussianErrorModelToBEAST.class,
               NegativeBinomialErrorModelToBEAST.class,
               PhyloDiscreteToBEAST.class
        );
    }

    @Override
    public List<Class<? extends ValueToBEAST>> getValuesToBEASTs() {
        return List.of(                
        CopyNumberBDToBEAST.class,
        IntegerCharacterMatrixToBEAST.class);
    }

    @Override
    public List<Class<? extends Generator>> getExcludedGenerator() {
        return List.of(
        CopyNumberBD.class,
        ReadCopyProfile.class);
    }

    @Override
    public Map<SequenceType, DataType> getDataTypeMap() {
        return Map.of();
    }

    @Override
    public List<Class> getExcludedValueType() {
        return List.of();
    }
}
