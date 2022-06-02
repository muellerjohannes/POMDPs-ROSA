using LinearAlgebra
using JuMP
using Ipopt
using Optim
using Plots
using Random
using Statistics
using DelimitedFiles

### Basic functions 
# Define infinite horizon discounted reward of a policy
function R(π, α, β, γ, μ, r)
    (nO, nS) = size(β)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * τ)
    Vπ = pinv(I - γ * transpose(pπ)) * (1-γ) * rπ
    reward = transpose(μ) * Vπ
    return reward
end

# Reward of a policy
function RExact(π, α, β, γ, μ, r)
    (nO, nS) = size(β)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * τ)
    Vπ = inv(I - γ * transpose(pπ)) * (1-γ) * rπ
    reward = transpose(μ) * Vπ
    return reward
end


### Bellman constrained programming 
function BCP(α, β, r, μ, γ, printLevel=4)
    (nO, nS) = size(β)
    nA = size(r)[2]
    # Define the model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    set_optimizer_attribute(model, "print_level", printLevel)
    # Introduce policy variables and constrain them to lie in the policy polytope
    @variable(model, π[1:nA, 1:nO]>=0);
    @constraint(model, [o in 1:nO], sum(π[:, o]) == 1);
    # Introduce the value function variables and add the Bellman constraint
    @variable(model, v[1:nS]);
    τ = π * β;
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS];
    rπ = diag(r * τ);
    @constraint(model, v .== (1-γ)*rπ + γ*transpose(pπ)*v);
    #Define the objective
    @NLobjective(model, Max, sum(v[s]*μ[s] for s in 1:nS))
    #Optimize
    optimize!(model)
    # Read of the policy obtained by Bellman constrained programming
    πOpt = JuMP.value.(π)
    # Return the optimal policy as well as the JuMP model
    return πOpt, model 
end

### Direct policy optimization 
# Define the tabular softmax policy parametrization
function softmaxPolicy(θ, nA, nO)
    θ = reshape(θ, (nA, nO))
    π = exp.(θ)
    for o in 1:nO
        π[:,o] = π[:,o] / sum(π[:,o])
    end
    return π
end

### Reward optimization in tate-action space 

# State action frequency for a policy
function stateActionFrequency(π, α, β, γ, μ, r)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    η = Diagonal(ρ) * transpose(τ)
    return η
end

# Linear equalities
function linearEqualities(η, μ, γ, α)
    nS = size(α)[1]
    linEq = [sum(η[s, :]) - γ*(dot(η[:, :], α[s, :, :])) - (1-γ)*μ[s] for s in 1:nS]
    if γ == 1
        linEq = append!(linEq, sum(η) - 1)
    end
    return linEq
end

# Polynomial equalities
function polynomialEqualities(η, β, deterministic=true)
    ### TODO: check whether beta is deterministic
    if sum(β.==1) < size(β)[2]
        error("method currently only implemented for deterministic observations")
    end
    nO = size(β)[1]
    polEq = []
    for o in 1:nO
        compatibleStates = findall(β[o,:] .> 0)
        sₒ = compatibleStates[1]
        Sₒ = setdiff(compatibleStates, sₒ)
        if length(Sₒ) > 0
            for s in Sₒ
                for a in 1:(nA-1)
                    polEq = vcat(polEq, η[s,a]*sum(η[sₒ,:]) - η[sₒ,a]*sum(η[s, :]))
                end
            end
        end
    end
    return polEq
end

# Compute the observation policy from the state action distribution
function observationPolicy(η, β)
    (nO, nS) = size(β)
    if rank(β) < nO
        error("observation mechanism does not satisfy the rank condition")
        return
    end
    nA = size(η)[2]
    # Compute the state policy
    τ = zeros((nS, nA));
    for i = 1:nS
        τ[i, :] = η[i, :] / sum(η[i, :]);
    end
    # Compute the observation policy
    π = transpose(τ) * pinv(β)
    return π
end

# Implementation of reward optimization in state-action space
function ROSA(α, β, r, μ, γ, printLevel=4)
    (nS, nA) = size(r)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    set_optimizer_attribute(model, "print_level", printLevel)
    @variable(model, η[1:nS,1:nA]>=0)
    lEqs = linearEqualities(η, μ, γ, α)
    pEqs = polynomialEqualities(η, β)
    h = vcat(lEqs, pEqs)
    for i in 1:length(h)
        p = h[i]
        @constraint(model, p == 0)
    end
    @NLobjective(model, Max, sum(η[s,a]*r[s,a] for a in 1:nA, s in 1:nS))
    optimize!(model)
    #Access the optimizer and compute the observation policy
    ηOpt = JuMP.value.(η)
    return ηOpt, model 
end


### Mazes 
# Storing the actions 
A = [-1  0; 0  -1; 1  0; 0  1]  # Actions corresponding to Up, Left, Down, Right
nA = 4

# Reading of the transition model α
# Finding all admissible states
function listOfStates(M)
    states = findall(M.>0)
    return states
end

# Automatically computation of the transition kernel; goal is the goal state, from which the
# agent transitions to the start state or to a uniform state if no start state is specified.
function transitionKernel(M,A,goal,start=false)
    states = findall(M .> 0)
    nS = length(states)
    α = zeros(nS,nS,nA)
    if !start
        for s in 1:nS
            for a in 1:nA
                x = states[s][1]
                y = states[s][2]
                if states[s] == goal 
                    α[:,s,:] = ones(nS, nA) / nS
                else
                    x += A[a,1]
                    y += A[a,2]
                    if M[x,y] == 1
                        snew = findall(i -> i == CartesianIndex(x,y), states)[1]
                        α[snew,s,a] = 1
                    else
                        α[s,s,a] = 1
                    end
                end
            end
        end
    elseif M[start] == 0
        error("start state is not admissible")
    else
        sₒ = findall(i -> i == start, states)
        for s in 1:nS
            for a in 1:nA
                x = states[s][1]
                y = states[s][2]
                if states[s] == goal  # CartesianIndex(x,y) in rew_positions
                    α[sₒ,s,:] = ones(nA)
                else
                    x += A[a][1]
                    y += A[a][2]
                    if M[x,y] == 1
                        sNew = findall(i -> i == CartesianIndex(x,y), states)[1]
                        α[sNew,s,a] = 1
                    else
                        α[s,s,a] = 1
                    end
                end
            end
        end
    end
    return α
end

# Reading of β for different observation types
# Returns the observation of state s under the observation filter V
function observe(M,s,V=[0 1 0; 1 0 1; 0 1 0])
    if (3,3) !== size(V)
        error("only 3x3 filters for the observations are supported")
    end
    states = listOfStates(M)
    if s > length(states)
        error("passed state is larger than the number of states")
    end
    observation = M[states[s][1]-1:states[s][1]+1, states[s][2]-1:states[s][2]+1].*V
    observation = reshape(observation, length(observation))
    observation = observation[reshape(V .!== 0, length(V))]
    return observation
end

# Returns a list of the occuring observations 
function listOfObservations(M, V=[0 1 0; 1 0 1; 0 1 0])
    states = listOfStates(M)
    nS = length(states)
    observations = map(s -> observe(M,s,V), 1:nS)
    observations = unique(observations)
    return observations
end

# Returns the observation kernel β
function observationKernel(M, V=[0 1 0; 1 0 1; 0 1 0])
    states = listOfStates(M)
    observations = listOfObservations(M, V)
    nS = length(states)
    nO = length(observations)
    β = zeros(nO, nS)
    for s in 1:nS
        β[findall(x -> x == observe(M,s,V) , observations)[1], s] = 1
    end
    return β
end

# Reading of the instantaneoues reward vector r
function instReward(M, A, goal)
    states = findall(M .> 0)
    r = map(i -> i == goal, states) * transpose(ones(nA)) * length(states)
    return r
end

# Generate initial distribution μ
function initialDistribution(M)
    states = listOfStates(M)
    nS = length(states)
    μ = ones(nS) / nS
    return μ
end

# Maze generation: modified from https://rosettacode.org/wiki/Maze_generation
check(bound::Vector) = cell -> all([1, 1] .≤ cell .≤ bound)
neighbors(cell::Vector, bound::Vector, step::Int=2) =
filter(check(bound), map(dir -> cell + step * dir, [[0, 1], [-1, 0], [0, -1], [1, 0]]))

function walk(maze::Matrix, nxtcell::Vector, visited::Vector=[])
    push!(visited, nxtcell)
    for neigh in shuffle(neighbors(nxtcell, collect(size(maze))))
        if neigh ∉ visited
            maze[round.(Int, (nxtcell + neigh) / 2)...] = 0
            walk(maze, neigh, visited)
        end
    end
    return maze
end
function maze(w::Int, h::Int)
    maze = collect(i % 2 | j % 2 for i in 1:2w+1, j in 1:2h+1)
    firstcell = 2 * [rand(1:w), rand(1:h)]
    M = walk(maze, firstcell)
    M = M .== 0
    return M
end

pprint(matrix) = for i = 1:size(matrix, 1) println(join(matrix[i, :])) end
function printmaze(maze)
    maze = maze.== 0
    walls = split("╹ ╸ ┛ ╺ ┗ ━ ┻ ╻ ┃ ┓ ┫ ┏ ┣ ┳ ╋")
    h, w = size(maze)
    f = cell -> 2 ^ ((3cell[1] + cell[2] + 3) / 2)
    wall(i, j) = if maze[i,j] == 0 " " else
        walls[Int(sum(f, filter(x -> maze[x...] != 0, neighbors([i, j], [h, w], 1)) .- [[i, j]]))]
    end
    mazewalls = collect(wall(i, j) for i in 1:2:h, j in 1:w)
    pprint(mazewalls)
end

