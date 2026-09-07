open module nestedBD.beast {
    requires beast.pkgmgmt;
    requires beast.base;

    requires org.apache.commons.statistics.distribution;  
    requires org.apache.commons.numbers.gamma;           

    exports nestedBD.evolution.errormodel;
    exports nestedBD.evolution.likelihood;
    exports nestedBD.evolution.substitutionmodel;

    provides beast.base.core.BEASTInterface with
        nestedBD.evolution.substitutionmodel.BD,
        nestedBD.evolution.likelihood.DiploidOriginLikelihood,
        nestedBD.evolution.likelihood.DiploidOriginLikelihoodWithError,
        nestedBD.evolution.likelihood.TreeLikelihoodWithError,
        nestedBD.evolution.errormodel.DiscreteGaussianErrorModel,
        nestedBD.evolution.errormodel.NegativeBinomialErrorModel,
        nestedBD.evolution.errormodel.poissonErrorModel,
        nestedBD.evolution.errormodel.NormalErrorModel,
        nestedBD.evolution.errormodel.readcountErrorModel;
}