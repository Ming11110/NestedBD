open module nestedBD.lphybeast {
    requires lphy.beast;
    requires nestedBD.beast;
    requires phylonco.lphy;
    requires beast.base;
    requires lphy.base;

    exports NestedBD.lphybeast.spi;
    exports NestedBD.lphybeast.tobeast.generators;
    exports NestedBD.lphybeast.tobeast.values;

    provides lphybeast.spi.LPhyBEASTMapping with NestedBD.lphybeast.spi.NestedBDLBImpl;
}
