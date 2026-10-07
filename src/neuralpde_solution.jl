"""
    get_phi_and_params(sol)

Return `(phi, θ)` for a neural Lyapunov `PDESystem` solved with NeuralPDE.jl, in the form
expected by [`get_numerical_lyapunov_function`](@ref) and [`get_policy`](@ref).

`sol` is the `SciMLBase.PDENoTimeSolution` returned by `solve` on the
`OptimizationProblem` produced by `NeuralPDE.discretize(pde_system, PhysicsInformedNN(...))`.

When the `PDESystem` has several dependent variables (`φ1`, `φ2`, ...), `phi` is a
`Vector` with one entry per dependent variable and `θ` is a `ComponentVector` with `θ[:φi]`
the parameters of `phi[i]`. When it has a single dependent variable, `phi` is a single
function and `θ` the flat parameter vector of its network.

Each entry of `phi` is called as `phi(x, θ)`: for a state vector `x` it returns a length-1
vector, and for a matrix whose columns are states it returns a `1 × n` matrix.
"""
function get_phi_and_params(sol::SciMLBase.PDENoTimeSolution)
    md = sol.disc_data
    opt_sol = sol.original_sol
    nets = Dict(nameof(net.depvar) => net for net in md.networks)
    names = Tuple(nameof(operation(Symbolics.value(dv))) for dv in md.pdesys.dvs)
    phi = [NeuralPDENetwork(nets[name]) for name in names]
    θs = [getu(opt_sol, nets[name].θ)(opt_sol) for name in names]
    return if length(phi) == 1
        only(phi), only(θs)
    else
        phi, ComponentArray(NamedTuple{names}(θs))
    end
end

struct NeuralPDENetwork{N}
    net::N
end

function (φ::NeuralPDENetwork)(x::AbstractVector, θ)
    return vec(NeuralPDE.trial_function(φ.net, θ)(reshape(x, :, 1)))
end
(φ::NeuralPDENetwork)(x::AbstractMatrix, θ) = NeuralPDE.trial_function(φ.net, θ)(x)
