using DataFrames, Combinatorics
include("HarmonicOscillatorWeak.jl")

function get_all_blocks(h::HarmonicOscillatorWeak; max_blocks = 0)
    max_blocks < 0 && throw(ArgumentError("max_blocks must be nonnegative"))
    S = h.S
    add0 = starting_address(h)
    N = num_particles(add0)

    df = DataFrame()

    block_id = 0
    known_basis = typeof(add0)[]
    tuples = with_replacement_combinations(1:prod(S), N)
    for t in tuples
        add = BoseFS(prod(S), (t .=> ones(Int, N))...)
        block_E0 = noninteracting_energy(h, add)
        if add in known_basis
            continue
        end

        block_id += 1
        block_E0 = noninteracting_energy(h, add)
        block_basis = BasisSetRep(h, add; sizelim = 1e10).basis;
        append!(known_basis, block_basis)
        push!(df, (; block_id, block_E0, block_size = length(block_basis), add))
        if max_blocks > 0 && block_id ≥ max_blocks
            break
        end
    end
    if max_blocks > 0
        return df
    elseif sum(df[!,:block_size]) == dimension(h)
        return df
    else
        error("not all blocks were found")
    end
end
