# HarmonicOscillatorCartesian
Hamiltonian for Rimu.jl - bosons with weak interactions in Cartesian harmonic oscillator basis

## Usage
Define parameters

    N = 3
    S = (6,6)
    η = 1
    g = 0.1

Dummy state to build Hamiltonian

    add0 = BoseFS(prod(S), 1 => N)
    H = HarmonicOscillatorWeak(add0; S, η, g, interaction_only = false)

Get all blocks

    block_df = get_all_blocks(H)
