"""
    get_all_blocks(h::HarmonicOscillatorWeak; 
        max_blocks = nothing, 
        target_energy = nothing
        method = :vertices,
        kwargs...) -> df

Find all distinct blocks of `h`. Returns a `DataFrame`. 

Keyword arguments:
* `max_blocks`: exit after finding this many blocks.
* `target_energy`: only blocks with this noninteracting energy are found. 
* `method`: Choose between `:vertices` and `:comb` for method of enumerating tuples of quantum numbers
* `save_to_file=nothing`: if set then the `DataFrame` recording blocks is saved after each new block is found
* additional `kwargs`: passed to `isapprox` for comparing block energies. Useful for anisotropic system.
"""
function get_all_blocks(h::HarmonicOscillatorWeak{D,P}; 
    target_energy = nothing, 
    max_energy = nothing, 
    max_blocks = nothing, 
    method = :vertices,
    kwargs...) where {D,P}

    add0 = starting_address(h)
    N = num_particles(add0)
    E0 = N * D / 2  # starting address may not be ground state
    if !isnothing(target_energy) && target_energy - E0 > minimum(h.S .* h.aspect) - 1
        @warn "target energy higher than single particle basis size; not all blocks may be found."
    end
    if !isnothing(max_energy) && max_energy < E0
        @warn "maximum requested energy lower than groundstate, not all blocks may be found."
    end
    if !isnothing(max_energy) && !isnothing(target_energy) && max_energy < target_energy
        @warn "maximum requested energy lower than target energy, not all blocks may be found."
    end

    if method == :vertices
        df = get_all_blocks_vertices(h; target_energy, max_energy, max_blocks, kwargs...)
    elseif method == :comb
        df = get_all_blocks_comb(h; target_energy, max_energy, max_blocks, kwargs...)
    else
        @error "invalid method."
    end

    # consistency check
    if isnothing(max_blocks) && isnothing(target_energy) && isnothing(max_energy) && sum(df[!,:block_size]) ≠ dimension(h)
        @warn "not all blocks were found"
    end
    return df
end

function get_all_blocks_vertices(h::HarmonicOscillatorWeak{D,P}; 
    target_energy = nothing, 
    max_energy = nothing, 
    max_blocks = nothing, 
    save_to_file = nothing,
    kwargs...) where {D,P}
    add0 = starting_address(h)
    N = num_particles(add0)

    # initialise
    df = DataFrame()
    block_id = 0
    known_basis = Set{typeof(add0)}()
    L = _binomial(P + N - 1, Val(N))
    idx_correction = reverse(ntuple(i -> i - 1, Val(N)))
    for i in 1:L
        t = vertices(i, Val(N)) .- idx_correction
        # check target energy
        block_E0 = noninteracting_energy(h, t)
        if !isnothing(target_energy)
            !isapprox(block_E0, target_energy; kwargs...) && continue
        end
        if !isnothing(max_energy)
            block_E0 > max_energy && continue
        end
        # check if known
        add = BoseFS(P, t .=> ones(Int, N))
        if add in known_basis
            continue
        end

        # new block found
        block_id += 1        
        block_basis = build_basis_only_from_LO(h, add)
        push!(known_basis, block_basis...)
        push!(df, (; block_id, block_E0, block_size = length(block_basis), add))
        !isnothing(save_to_file) && save_df(save_to_file, df)
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    return df
end

# old version - issues with GC due to allocating many small vectors
function get_all_blocks_comb(h::HarmonicOscillatorWeak{D,P}; 
    target_energy = nothing, 
    max_energy = nothing, 
    max_blocks = nothing, 
    save_to_file = nothing,
    kwargs...) where {D,P}
    add0 = starting_address(h)
    N = num_particles(add0)

    # initialise
    df = DataFrame()
    block_id = 0
    known_basis = Set{typeof(add0)}()
    tuples = with_replacement_combinations(1:P, N)
    for t in tuples
        # check target energy
        block_E0 = noninteracting_energy(h, t)
        if !isnothing(target_energy)
            !isapprox(block_E0, target_energy; kwargs...) && continue
        end
        if !isnothing(max_energy)
            block_E0 > max_energy && continue
        end
        # check if known
        add = BoseFS(P, t .=> ones(Int, N))
        if add in known_basis
            continue
        end

        # new block found
        block_id += 1        
        block_basis = build_basis_only_from_LO(h, add)
        push!(known_basis, block_basis...)
        push!(df, (; block_id, block_E0, block_size = length(block_basis), add))
        !isnothing(save_to_file) && save_df(save_to_file, df)
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    return df
end

"""
    pick_starting_state(E, N, dims)

Convenience function for picking an `N` boson state with desired total energy `E`.
Assumes energy gap is the same in all dimensions and ignores groundstate energy.
"""
function pick_starting_state(E_target, N, dims::NTuple{D,Int}) where {D}
    @assert 0 ≤ E_target ≤ maximum(dims) * N
    modes = zeros(Int, N)
    
    # something like this:
    # Still need to properly account for different dimensions
    for n in 1:N, d in 1:D
        if E_target < dims[d]
            break
        else
            E_target -= dims[d]
            modes[n] += dims[d]
        end
    end
    return BoseFS(prod(dims), [modes[n] + 1 => 1 for n in 1:N]...)
end

"""
    fock_to_cartHO_basis(basis, S)

Convert all Fock states in `basis` to Cartesian harmonic oscillator basis
indices (nx,ny,...), bounded by `S`, and print.
"""
function fock_to_cartHO_basis(basis, S)
    @assert all(prod(S) .== num_modes.(basis))
    states = CartesianIndices(S)    

    cart = map(
            add -> vcat(map(p -> [Tuple(states[p.mode]) .- 1 for _ in 1:p.occnum], OccupiedModeMap(add))...),
            basis)

    return cart
end