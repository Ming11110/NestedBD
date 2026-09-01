package NestedBD.lphybeast.spi;

import beast.base.evolution.datatype.DataType;
import jebl.evolution.sequences.SequenceType;
import lphy.core.model.Generator;
import lphybeast.GeneratorToBEAST;
import lphybeast.ValueToBEAST;
import lphybeast.spi.LPhyBEASTMapping;

import NestedBD.lphybeast.tobeast.generators.DiscreteGaussianErrorModelToBEAST;
import NestedBD.lphybeast.tobeast.generators.NegativeBinomialErrorModelToBEAST;
import NestedBD.lphybeast.tobeast.generators.PhyloDiscreteToBEAST;
import NestedBD.lphybeast.tobeast.values.CopyNumberBDToBEAST;
import NestedBD.lphybeast.tobeast.values.IntegerCharacterMatrixToBEAST;

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
        return List.of();
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
