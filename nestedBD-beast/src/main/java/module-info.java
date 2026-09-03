open module nestedBD.beast {
    requires beast.pkgmgmt;
    requires beast.base;

    requires org.apache.commons.statistics.distribution;  
    requires org.apache.commons.numbers.gamma;           

    exports NestedBD.evolution.errormodel;
    exports NestedBD.evolution.likelihood;
    exports NestedBD.evolution.substitutionmodel;

    provides beast.base.core.BEASTInterface with
        NestedBD.evolution.substitutionmodel.BD,
        NestedBD.evolution.likelihood.DiploidOriginLikelihood,
        NestedBD.evolution.likelihood.DiploidOriginLikelihoodWithError,
        NestedBD.evolution.likelihood.TreeLikelihoodWithError,
        NestedBD.evolution.errormodel.DiscreteGaussianErrorModel,
        NestedBD.evolution.errormodel.NegativeBinomialErrorModel,
        NestedBD.evolution.errormodel.poissonErrorModel,
        NestedBD.evolution.errormodel.NormalErrorModel,
        NestedBD.evolution.errormodel.readcountErrorModel;
}