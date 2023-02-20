# TO-DO: optimise if large arguments are used
"""
    delta_interaction_matrix_element(i, j, k, l; max_level = typemax(Int))

Integral of four one-dimensional harmonic oscillator functions. It is expected that this function
should be used when energy is conserved, i.e. ``i+j == k+l``, but this is not enforced.
State indices start at `0` for the groundstate.
"""
function delta_interaction_matrix_element(i, j, k, l; max_level = typemax(Int))
    if !all(0 .≤ (i,j,k,l) .≤ max_level)
        return 0.0
    end
    min_kl = min(k, l)
    # SpecialFunctions.gamma(x) is faster than factorial(big(x))
    p = sqrt(gamma(k + 1) * gamma(l + 1)) / sqrt(2 * gamma(i + 1) * gamma(j + 1)) / pi^2
    s = sum(gamma(t + 1/2) * gamma(i - t + 1/2) * gamma(j - t + 1/2) / (gamma(t + 1) * gamma(k - t + 1) * gamma(l - t + 1)) for t in 0 : min_kl)
    
    return p * s / 2
end

"""
    corners(dims::NTuple{D,Int})

Enumerate corners of a box with side lengths `dims`.
"""
function corners(dims::NTuple{D,Int}) where {D}
    rngs = ntuple(d -> 1:dims[d]-1:dims[d], Val(D))
    return CartesianIndices(rngs)
end

"""
    largest_two_point_interval(i::Int, j::Int, w::Int) -> ranges

For two points `i` and `j` on a range `1:w`, find the largest subinterval of sites 
such that moving `i` to one of those sites and moving `j` by an equal but opposite
amount leaves both within `1:w`.
"""
function largest_two_point_interval(i::Int, j::Int, w::Int)
    if i ≤ j
        left_gap = min(i - 1, w - j)
        left_edge = i - left_gap
        right_gap = min(w - i, j - 1)
        right_edge = i + right_gap
    else
        left_gap = min(j - 1, w - i)
        left_edge = j - left_gap
        right_gap = min(w - j, i - 1)
        right_edge = j + right_gap
    end
    return left_edge:right_edge
end

"""
    largest_two_point_box(i, j, dims::Tuple) -> ranges

For a box defined by `dims` containing two points `i` and `j`, find the set of all positions 
such that shifting `i` to that position with an opposite shift to `j` leaves both points inbounds.

Arguments `i` and `j` are linear indices, `dims` is a `Tuple` defining the inbounds area.
"""
function largest_two_point_box(i::Int, j::Int, dims::NTuple{D,Int}) where {D}
    cart = CartesianIndices(dims)
    state_i = Tuple(cart[i])

    state_j = Tuple(cart[j])

    ranges = ntuple(d -> largest_two_point_interval(state_i[d], state_j[d], dims[d]), Val(D))
    
    box_size = prod(length.(ranges))
    return ranges, box_size
end

# REDUNDANT
"""
    get_bose_pairs(omm::BoseOccupiedModeMap)

Enumerate pairs of bosons in `omm`.
"""
function get_bose_pairs(omm::BoseOccupiedModeMap)
    s = length(omm)
    # d = count(i -> i.occnum ≥ 2, omm)
    # numpairs = d + binomial(s, 2)     # only useful if I can add pairs without pushing
    bosepairs = Tuple{BoseFSIndex,BoseFSIndex}[]

    for i in 1:s
        particle_i = omm[i]
        occ_i = omm[i].occnum
        # mode_i = omm[i].mode
        if occ_i > 1
            # use i in 1:M indexing for accessing table            
            push!(bosepairs, (particle_i, particle_i))
        end
        for j in 1:i-1
            particle_j = omm[j]
            # occ_j = omm[j].occnum
            # mode_j = omm[j].mode
            push!(bosepairs, (particle_i, particle_j))
        end
    end
    
    return bosepairs
end

"""
    find_chosen_pair_moves(omm::OccupiedModeMap, c, S) -> p_i, p_j, c, box_ranges

Find size of valid moves for chosen pair of indices in `omm`. Returns two valid indices `p_i` and `p_j`
for the initial pair and a tuple of ranges `box_ranges` defining the subbox of valid moves. 
The index for the chosen move `c` is updated to be valid for this box.
Arguments are the `OccupiedModeMap` `omm`, the chosen move index `c`
and the size of the basis grid `S`.
"""
function find_chosen_pair_moves(omm::OccupiedModeMap, c, S::Tuple)
    for i in 1:length(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            box_ranges, box_size = largest_two_point_box(p_i.mode, p_i.mode, S)
            if c - box_size < 1
                return p_i, p_i, box_ranges, c
            else
                c -= box_size 
            end
        end
        for j in 1:i-1
            p_j = omm[j]
            box_ranges, box_size = largest_two_point_box(p_i.mode, p_j.mode, S)
            if c - box_size < 1
                return p_i, p_j, box_ranges, c
            else
                c -= box_size 
            end
        end
    end
    error("chosen pair not found")
end

"""
    HarmonicOscillatorWeak(add; S = (num_modes(add),), η = ones(D), g = 1.0, interaction_only = false)

Implements a one-dimensional harmonic oscillator in harmonic oscillator basis with weak interactions.

```math
\\hat{H} = \\sum_{i} ϵ_i n_i + \\frac{g}{2}\\sum_{ijkl} V_{ijkl} a^†_i a^†_j a_k a_l δ_{i+j,k+l}
```

# Arguments

* `add`: the starting address, defines number of particles and total number of modes.
* `S`: Defines the number of levels in each dimension, including the groundstate. Defaults 
    to a 1D spectrum with number of levels matching modes of `add`. 
* `η`: The aspect ratios of the trap. For anisotropic traps the length of `η` should be the 
    number of dimensions `D`, the same as the length of `S`. Aspect ratios are scaled relative 
    to the first dimension, which sets the energy scale of the system, ``\\hbar\\omega_x``.
* `g`: the (isotropic) interparticle interaction parameter. The value of `g` is assumed to be in trap units.
* `interaction_only`: if set to `true` then the noninteracting trap energies are set to zero.
"""
struct HarmonicOscillatorWeak{
    D,  # number of dimensions
    P,  # total states
    M,  # size of largest dimension
    A<:BoseFS
} <: AbstractHamiltonian{Float64}
    add::A
    S::NTuple{D,Int64}
    aspect::NTuple{D,Float64}
    energies::Vector{Float64} # noninteracting single particle energies
    vtable::Array{Float64,3}  # interaction coefficients
    u::Float64
end

function HarmonicOscillatorWeak(
        add::BoseFS; 
        S::NTuple{D,Int64} = (num_modes(add),),
        η = 1.0, 
        g = 1.0,
        interaction_only = false,
        debug_compilation = false
    ) where {D}
    
    P = prod(S)
    @assert P == num_modes(add)

    if length(η) == D
        # aspect = float([η...]) ./ η[1]
        aspect = ntuple(i -> η[i] / η[1], Val(D))
    elseif η == 1.0
        # aspect = ones(D)
        aspect = ntuple(i -> 1.0, Val(D))
    else
        throw(ArgumentError("Invalid aspect ratio parameter η."))
    end

    if interaction_only
        energies = SVector{P}(zeros(P))
    else
        states = CartesianIndices(S)    # 1-indexed
        # energies = map(x -> dot(aspect, Tuple(x) .- 1/2), states)
        energies = reshape(map(x -> dot(aspect, Tuple(x) .- 1/2), states), P)
    end

    u = sqrt(prod(aspect)) * g / 2

    M = maximum(S)
    bigrange = 0:M-1
    # case n = M is the same as n = 0
    # vmat = SArray{Tuple{M,M,M}}([
    #         delta_interaction_matrix_element(i-n, j+n, j, i; max_level = M-1)
    #             for i in bigrange, j in bigrange, n in bigrange]
    #             )

    vmat = if debug_compilation
            delta_interaction_matrix_element(1,1,1,1; max_level = 1)
            delta_interaction_matrix_element(1,1,1,1; max_level = 0)
            zeros(M,M,M)
        else
            reshape(
                [delta_interaction_matrix_element(i-n, j+n, j, i; max_level = M-1) 
                    for i in bigrange, j in bigrange, n in bigrange],
                    M,M,M)
        end

    return HarmonicOscillatorWeak{D,P,M,typeof(add)}(add, S, aspect, energies, vmat, u)
end

function Base.show(io::IO, h::HarmonicOscillatorWeak)
    print(io, "HarmonicOscillatorWeak($(h.add); S=$(h.S), η=$(h.aspect), u=$(h.u))")
end

function starting_address(h::HarmonicOscillatorWeak)
    return h.add
end

LOStructure(::Type{<:HarmonicOscillatorWeak}) = IsHermitian()

# Base.getproperty(h::HarmonicOscillatorWeak, s::Symbol) = getproperty(h, Val(s))
# Base.getproperty(h::HarmonicOscillatorWeak, ::Val{:ks}) = getfield(h, :ks)
# Base.getproperty(h::HarmonicOscillatorWeak, ::Val{:kes}) = getfield(h, :kes)
# Base.getproperty(h::HarmonicOscillatorWeak, ::Val{:add}) = getfield(h, :add)
# Base.getproperty(h::HarmonicOscillatorWeak, ::Val{:vtable}) = getfield(h, :vtable)
# Base.getproperty(h::HarmonicOscillatorWeak{<:Any,<:Any,<:Any,<:Any,U}, ::Val{:u}) where {U} = U
# Base.getproperty(h::HarmonicOscillatorWeak{<:Any,<:Any,<:Any,W}, ::Val{:w}) where {W} = W
# Base.getproperty(h::HarmonicOscillatorWeak{<:Any,D}, ::Val{:dim}) where {D} = D


### DIAGONAL ELEMENTS ###
function energy_transfer_diagonal(h::HarmonicOscillatorWeak{D}, omm::BoseOccupiedModeMap) where {D}
    result = 0.0
    states = CartesianIndices(h.S)    # 1-indexed
    
    for i in 1:length(omm)
        mode_i, occ_i = omm[i].mode, omm[i].occnum
        idx_i = Tuple(states[mode_i])
        if occ_i > 1
            # use i in 1:M indexing for accessing table
            val = occ_i * (occ_i - 1)
            result += prod(h.vtable[idx_i[d],idx_i[d],1] for d in 1:D) * val
        end
        for j in 1:i-1
            mode_j, occ_j = omm[j].mode, omm[j].occnum
            idx_j = Tuple(states[mode_j])
            val = 4 * occ_i * occ_j
            result += prod(h.vtable[idx_i[d],idx_j[d],1] for d in 1:D) * val
        end        
    end
    return result * h.u
end

noninteracting_energy(h::HarmonicOscillatorWeak, omm::BoseOccupiedModeMap) = dot(h.energies, omm)
@inline function noninteracting_energy(h::HarmonicOscillatorWeak, add::BoseFS)
    omm = OccupiedModeMap(add)
    return noninteracting_energy(h, omm)
end
# fast method for finding blocks
noninteracting_energy(h::HarmonicOscillatorWeak, t::Union{Vector{Int64},NTuple{N,Int64}}) where {N} = sum(h.energies[j] for j in t)

@inline function diagonal_element(h::HarmonicOscillatorWeak, add::BoseFS)
    omm = OccupiedModeMap(add)
    return noninteracting_energy(h, omm) + energy_transfer_diagonal(h, omm)
end

### OFFDIAGONAL ELEMENTS ###

# includes swap moves and trivial moves
# To-Do: optimise these out for FCIQMC
function num_offdiagonals(h::HarmonicOscillatorWeak, add::BoseFS)
    S = h.S
    omm = OccupiedModeMap(add)
    noffs = 0

    for i in 1:length(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            _, valid_box_size  = largest_two_point_box(p_i.mode, p_i.mode, S)
            noffs += valid_box_size 
        end
        for j in 1:i-1
            p_j = omm[j]
            _, valid_box_size  = largest_two_point_box(p_i.mode, p_j.mode, S)
            noffs += valid_box_size 
        end
    end
    return noffs
end


"""
    energy_transfer_offdiagonal(S, add, chosen, omm = OccupiedModeMap(add))
        -> new_add, val, mode_i, mode_j, mode_l

Return the new address `new_add`, the prefactor `val`, the initial particle modes
`mode_i` and `mode_j` and the new mode for `i`th particle, `mode_l`. The other new mode
`mode_k` is implicit by energy conservation.
"""
function energy_transfer_offdiagonal(
        S::Tuple, 
        add::BoseFS, 
        chosen::Int, 
        omm::BoseOccupiedModeMap = OccupiedModeMap(add)
    )
    # find size of valid moves for each pair
    particle_i, particle_j, valid_box_ranges, chosen = find_chosen_pair_moves(omm, chosen, S)
    mode_i = particle_i.mode
    mode_j = particle_j.mode

    # This is probably not optimal
    mode_l = LinearIndices(CartesianIndices(S))[CartesianIndices(valid_box_ranges)[chosen]]
    # discard swap moves and self moves
    if mode_l == mode_j || mode_l == mode_i
        return add, 0.0, 0, 0, 0
    end
    mode_Δn = mode_l - mode_i
    mode_k = mode_j - mode_Δn
    particle_k = find_mode(add, mode_k)
    particle_l = find_mode(add, mode_l)

    new_add, val = excitation(add, (particle_l, particle_k), (particle_j, particle_i))

    return new_add, val, mode_i, mode_j, mode_l
end

function get_offdiagonal(
        h::HarmonicOscillatorWeak{D,<:Any,<:Any,A}, 
        add::A, 
        chosen::Int, 
        omm::BoseOccupiedModeMap = OccupiedModeMap(add)
    ) where {D,A}

    S = h.S
    states = CartesianIndices(S)    # 1-indexed
    newadd, val, i, j, l = energy_transfer_offdiagonal(S, add, chosen, omm)

    if val ≠ 0.0    # is this a safe check? maybe check if Δns == (0,...)
        idx_i = Tuple(states[i])
        idx_j = Tuple(states[j])
        idx_l = Tuple(states[l])

        # sort indices to match table of values
        idx_i_sort = ntuple(d -> idx_i[d] > idx_l[d] ? idx_i[d] : idx_j[d], Val(D))
        idx_j_sort = ntuple(d -> idx_i[d] > idx_l[d] ? idx_j[d] : idx_i[d], Val(D))
        idx_Δns = ntuple(d -> abs(idx_i[d] - idx_l[d]) + 1, Val(D))

        result = prod(h.vtable[a,b,c] for (a,b,c) in zip(idx_i_sort, idx_j_sort, idx_Δns)) * val

        # account for swap of (i,j)
        result *= 1 + (i ≠ j)
    else 
        result = 0.0
    end
    return newadd, result * h.u
end

###
### offdiagonals
###
"""
    HOWeakOffdiagonals

Specialized [`AbstractOffdiagonals`](@ref) that keeps track of singly and doubly occupied
sites in current address.
"""
struct HOWeakOffdiagonals{
    A<:BoseFS,T,H<:AbstractHamiltonian{T},O<:OccupiedModeMap
} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
    map::O
end

function offdiagonals(h::HarmonicOscillatorWeak, add::BoseFS)
    omm = OccupiedModeMap(add)
    num = num_offdiagonals(h, add)
    return HOWeakOffdiagonals(h, add, num, omm)
end

function Base.getindex(s::HOWeakOffdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck begin
        1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    end
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i, s.map)
    return (new_address, matrix_element)
end

Base.size(s::HOWeakOffdiagonals) = (s.length,)
