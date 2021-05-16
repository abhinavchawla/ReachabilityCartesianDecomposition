# ReachabilityCartesianDecomposition
An implementation of the paper "Reach set approximation through decomposition with low-dimensional sets and high-dimensional matrices" in Python using the zonotope set representation. This was done as the final project of CSE510- Hybrid Systems and Trusted Autonomy

This is a faster way to approximate reach set approximations where the initial set is decomposed into multiple smaller sets and we try to create a reachability analysis on these smaller sets while exponentiating the high dimensional phi matrix at every step. This implementation uses the zonotope as the set representation and is implemented in Python. An example of a harmonic oscillator system using this method is shown.
<img width="598" alt="image" src="https://user-images.githubusercontent.com/9547429/118386529-08d1ba00-b5e6-11eb-887c-59cd766aeb6b.png">
