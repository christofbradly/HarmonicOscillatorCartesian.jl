module HarmonicOscillatorCartesian

using StaticArrays
using LinearAlgebra
using SpecialFunctions
using DataFrames, Combinatorics

using Rimu
using Rimu.Hamiltonians: num_singly_doubly_occupied_sites, AbstractOffdiagonals, BoseOccupiedModeMap
import Rimu.Interfaces: starting_address, num_offdiagonals, get_offdiagonal, offdiagonals, diagonal_element

export HarmonicOscillatorWeak
export get_all_blocks, get_all_blocks_vertices, fock_to_cartHO_basis

include("HarmonicOscillatorWeak.jl")
include("vertices.jl")
include("blocks.jl")

end