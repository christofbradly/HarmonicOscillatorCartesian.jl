"""
    get_all_blocks(h::HarmonicOscillatorWeak; max_blocks = 0, target_energy = nothing) -> df

Find all distinct blocks of `h`. Returns a `DataFrame`. 

If `target_energy` is set then only blocks with the same noninteracting energy are found.
Keyword arguments are passed to `isapprox` for comparing these energies.
If `max_blocks` is set then the loop over all basis states will be interrupted after 
`max_blocks` have been found.
"""
function get_all_blocks(h::HarmonicOscillatorWeak{D,P}; 
        target_energy = nothing, 
        max_blocks = nothing, 
        kwargs...) where {D,P}
    add0 = starting_address(h)
    N = num_particles(add0)
    if !isnothing(target_energy)
        # starting address may not be ground state
        E0 = N * D / 2
        if target_energy - E0 > minimum(h.S) - 1
            @warn "target energy higher than single particle basis size; not all blocks will be found"
        end
    end

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
        # check if known
        add = BoseFS(P, t .=> ones(Int, N))
        if add in known_basis
            continue
        end

        # new block found
        block_id += 1        
        block_basis = BasisSetRep(h, add; sizelim = 1e10).basis;
        push!(known_basis, block_basis...)
        push!(df, (; block_id, block_E0, block_size = length(block_basis), add))
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    if !isnothing(max_blocks) || !isnothing(target_energy) || sum(df[!,:block_size]) == dimension(h)
        return df
    else
        error("not all blocks were found")
    end
end

function get_all_blocks_vertices(h::HarmonicOscillatorWeak{D,P}; 
    target_energy = nothing, 
    max_blocks = nothing, 
    kwargs...) where {D,P}
    add0 = starting_address(h)
    N = num_particles(add0)
    if !isnothing(target_energy)
        # starting address may not be ground state
        E0 = N * D / 2
        if target_energy - E0 > minimum(h.S) - 1
            @warn "target energy higher than single particle basis size; not all blocks will be found"
        end
    end

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
        # check if known
        add = BoseFS(P, t .=> ones(Int, N))
        if add in known_basis
            continue
        end

        # new block found
        block_id += 1        
        block_basis = BasisSetRep(h, add; sizelim = 1e10).basis;
        push!(known_basis, block_basis...)
        push!(df, (; block_id, block_E0, block_size = length(block_basis), add))
        if !isnothing(max_blocks) && block_id ≥ max_blocks
            break
        end
    end
    if !isnothing(max_blocks) || !isnothing(target_energy) || sum(df[!,:block_size]) == dimension(h)
        return df
    else
        error("not all blocks were found")
    end
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