# calculate the basis without storing the sparse matrix
# remove once this is added to Rimu
function build_basis_only_from_LO(
    ham::HarmonicOscillatorWeak, address=starting_address(ham);
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    sort=false, kwargs...,
)
    if !isnothing(filter) && !filter(address)
        throw(ArgumentError(string(
            "Starting address does not pass `filter`. ",
            "Please pick a different address or a different filter."
        )))
    end
    adds = [address]          # Queue of addresses. Also returned as the basis.
    dict = Dict(address => 1) # Mapping from addresses to indices

    i = 0
    while i < length(adds)
        i += 1
        add = adds[i]

        for (off, v) in offdiagonals(ham, add)
            iszero(v) && continue
            j = get(dict, off, nothing)
            if isnothing(j)
                # Energy cutoff: remember skipped addresses, but avoid adding them to `adds`
                if !isnothing(filter) && !filter(off)
                    dict[off] = 0
                    j = 0
                else
                    push!(adds, off)
                    j = length(adds)
                    dict[off] = j
                end
            end
        end
    end

    if sort
        perm = sortperm(adds; kwargs...)
        return permute!(adds, perm)
    else
        return adds
    end
end