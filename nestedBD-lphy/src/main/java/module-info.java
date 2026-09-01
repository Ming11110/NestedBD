/**
 * @author Walter Xie
 */
module phylonco.lphy {
    requires transitive lphy.base;
    requires jdk.jfr;

    //copy number model
    exports phylonco.lphy.evolution.copynumbermodel;

    // declare what service interface the provider intends to use
    uses lphy.core.spi.Extension;
    provides lphy.core.spi.Extension with phylonco.lphy.spi.NestedBDImpl;
}