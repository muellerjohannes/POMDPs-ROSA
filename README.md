# Solving infinite-horizon POMDPs with memoryless stochastic policies in state-action space

This repo decoments the code used for the experiments presented in the extended abstract *Solving infinite-horizon POMDPs with memoryless stochastic policies in state-action space* presented at RLDM 2022 (see https://arxiv.org/abs/2205.14098). This includes an implementation of the presented method for reward optimization in state-action space (ROSA) as well as the two baselines used for comparison.






We provide code, which solves POMDPs in state action space by solving the polynomially constrained optimization problem with linear objective associated to the POMDP. As a reference, we also implement Bellman constrained programming and apply L-BFGS directly to the reward function for tabular softmax policies.

* In utilitiesGeneral.jl there are implementations of the infinite horizon discounted reward function, the tabular softmax parametrization and Bellman constrained programming (BCP). It requires the transition matrices of the POMDP, the discout factor, intitial distribution and instantatneous reward vector.
* In utilitiesStateAction.jl are implementations regarding computations in state-action space. This includes the computation of the linear and polynomial constraints for deterministic observations as well as the computation of an observation policy from a state-action frequency. The function ROSA implements the reward optimization in state-action space. It requires the transition matrices of the POMDP, the discout factor, intitial distribution and instantatneous reward vector.
* In utilitiesMazes.jl we provide code that automatically generates the transition matrices associated with a maze. It requires the maze as a two dimensional 0-1 array as well the observation mask, which specifies, which neighboring states can be observed. Further, it generates the instantaneous reward vector, which corresponds to a reward obtained at a randomly chosen state. Finally, we provide code generating random mazes of given size, which is adapted from ....

We demonstrate the application of the three different approaches to solve a navigation problem in a random maze in example.jl and in the Jupyter notebook example....
