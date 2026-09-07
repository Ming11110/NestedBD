/**
 * @author Walter Xie
 */
module nestedBD.lphy {
    requires transitive lphy.base;
    requires jdk.jfr;

    //copy number model
    exports nestedBD.lphy.evolution.copynumbermodel;

    // declare what service interface the provider intends to use
    uses lphy.core.spi.Extension;
    provides lphy.core.spi.Extension with nestedBD.lphy.spi.NestedBDImpl;
}