"""

    NeuralLyapunovBenchmarkLogger(losses, iterations)

A simple logger for tracking the (full weighted) loss during training, filled by the
Optimization.jl callback returned by [`NeuralLyapunov.logging_callback`](@ref).

# Fields
  - `losses::Vector{<:Real}`: A vector to store the loss values.
  - `iterations::Vector{<:Integer}`: A vector to store the corresponding iteration numbers.

# Constructors
  - `NeuralLyapunovBenchmarkLogger{T1, T2}() where {T1<:Real, T2<:Integer}`: Creates an
    empty logger with specified types for losses and iterations.
  - `NeuralLyapunovBenchmarkLogger{T}() where {T}`: Creates an empty logger with specified
    type for losses and `Int64` for iterations.
  - `NeuralLyapunovBenchmarkLogger()`: Creates an empty logger with `Float64` for losses and
    `Int64` for iterations.
"""
struct NeuralLyapunovBenchmarkLogger{T1 <: Real, T2 <: Integer}
    losses::Vector{T1}
    iterations::Vector{T2}
end

function NeuralLyapunovBenchmarkLogger{T1, T2}() where {T1, T2}
    return NeuralLyapunovBenchmarkLogger{T1, T2}(T1[], T2[])
end
NeuralLyapunovBenchmarkLogger{T}() where {T} = NeuralLyapunovBenchmarkLogger{T, Int64}()
NeuralLyapunovBenchmarkLogger() = NeuralLyapunovBenchmarkLogger{Float64}()

"""
    logging_callback(logger, log_frequency)

Return an Optimization.jl callback `(state, loss) -> false` that records `loss` in `logger`
every `log_frequency` calls. The iteration count continues across every optimization run
that uses the returned callback.
"""
function logging_callback(logger::NeuralLyapunovBenchmarkLogger, log_frequency::Integer)
    iteration = Ref(0)
    return function (state, loss, args...)
        iteration[] += 1
        if iteration[] % log_frequency == 0
            push!(logger.losses, loss)
            push!(logger.iterations, iteration[])
        end
        return false
    end
end
