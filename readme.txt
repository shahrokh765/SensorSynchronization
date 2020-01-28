This project has been done to use the sensors reading(power values) in situation where sensors have a clock skew among each other so creating Observation Vector(O) may lead to error when we want to process their data for different purposes(e.g. Localization of Intruders(or TXs)).
The only information you should provide is the maximum possible skew between any pair of sensors.
The idea is creating a DAG with vertices(group) that represents a set of sensors such that they receive power from the same set of TXs and an edge e=(u, v) show group u hear from a set of TXs which is a subset of TXs that group v can receive signal from. Please refer to the IEEE DySPAN'19 conference paper for more technical and theorical information.

Title is "Multiple Transmitter Localization under Time-Skewed Observations"
https://ieeexplore.ieee.org/document/8935739
 
