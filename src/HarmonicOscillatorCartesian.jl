module HarmonicOscillatorCartesian
    using Rimu
    using Rimu.Hamiltonians: num_singly_doubly_occupied_sites, AbstractOffdiagonals, BoseOccupiedModeMap
    import Rimu.Interfaces: starting_address, num_offdiagonals, get_offdiagonal, offdiagonals, diagonal_element
    using StaticArrays
    using LinearAlgebra
    using SpecialFunctions
    using DataFrames, Combinatorics

    export HarmonicOscillatorWeak
    export get_all_blocks

    include("HarmonicOscillatorWeak.jl")
    include("blocks.jl")
end