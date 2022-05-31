# Solving infinite-horizon POMDPs with memoryless stochastic policies in state-action space

This repo decoments the code used for the experiments presented in the extended abstract *Solving infinite-horizon POMDPs with memoryless stochastic policies in state-action space* presented at RLDM 2022 (see https://arxiv.org/abs/2205.14098 and also https://arxiv.org/abs/2110.07409 for theoretical discussion of the geometry of the optimization problem). This includes an implementation of the presented method for reward optimization in state-action space (ROSA) as well as the two baselines used for comparison.

Overview over the content:

* utilities.jl: Containts implementations of basic functions like the reward as well as implementations of the solution of the Bellmann constrained program (BCP) proposed by Amato et. al. (see http://people.csail.mit.edu/camato/publications/OptimalPOMDP-aimath05.pdf) as well as the reward optimization in state-action space (ROSA), both relying on the interior point solver IPOpt (see https://coin-or.github.io/Ipopt/).

* 
