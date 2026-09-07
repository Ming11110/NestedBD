open module nestedBD.lphybeast {
    requires lphy.beast;
    requires nestedBD.beast;
    requires nestedBD.lphy;
    requires beast.base;
    requires lphy.base;

    exports nestedBD.lphybeast.spi;
    exports nestedBD.lphybeast.tobeast.generators;
    exports nestedBD.lphybeast.tobeast.values;

    provides lphybeast.spi.LPhyBEASTMapping with nestedBD.lphybeast.spi.NestedBDLBImpl;
}
