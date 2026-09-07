package nestedBD.lphybeast.tobeast.values;

import nestedBD.evolution.substitutionmodel.BD;
import beast.base.spec.inference.parameter.IntScalarParam;
import beast.base.spec.domain.PositiveInt;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.domain.PositiveReal;

import lphy.core.model.Value;
import lphybeast.BEASTContext;
import lphybeast.ValueToBEAST;
import nestedBD.lphy.evolution.copynumbermodel.CopyNumberBD;

public class CopyNumberBDToBEAST implements ValueToBEAST<CopyNumberBD, BD> {

    @Override
    public BD valueToBEAST(Value<CopyNumberBD> value, BEASTContext context) {
        CopyNumberBD copyNumberBD = value.value();

        // Create the BEAST BD model
        BD bdModel = new BD();

        // Get nstates
        int nstates = copyNumberBD.getNstate().value();
        IntScalarParam<PositiveInt> nstateParam = new IntScalarParam<>(nstates, PositiveInt.INSTANCE);

        // Get bdRate from the LPhy model
        double bdRate = copyNumberBD.getBdRate().value();
        RealScalarParam<PositiveReal> bdRateParam = new RealScalarParam<>(bdRate, PositiveReal.INSTANCE);

        // Set inputs of BD model
        bdModel.setInputValue("nstate", nstateParam);
        bdModel.setInputValue("bdRate", bdRateParam);
        bdModel.initAndValidate();

        // Return the completed BD model
        return bdModel;
    }

    @Override
    public Class getValueClass() {
        return CopyNumberBD.class;
    }

    @Override
    public Class<BD> getBEASTClass() {
        return BD.class;
    }
}
