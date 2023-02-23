# calculate the basis without storing the sparse matrix
# remove once this is added to Rimu
function build_basis(
    ham::HarmonicOscillatorWeak, address=starting_address(ham);
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    sort=false, 
    max_size=Inf, 
    kwargs...,
)
    # check_address_type(ham, address)
    if !isnothing(filter) && !filter(address)
        throw(ArgumentError(string(
            "Starting address does not pass `filter`. ",
            "Please pick a different address or a different filter."
        )))
    end
    dimension(Float64, ham) < max_size || throw(ArgumentError("dimension larger than max_size"))
    adds = [address]        # Queue of addresses. Also returned as the basis.
    known_basis = Set(adds)     # known addresses

    i = 0
    while i < length(adds)
        i += 1
        add = adds[i]

        for (off, v) in offdiagonals(ham, add)
            (iszero(v) || off in known_basis) && continue   # check if valid
            push!(known_basis, off)
            !isnothing(filter) && !filter(off) && continue  # check filter
            push!(adds, off)
        end
    end

    if sort
        return sort!(adds, kwargs...)
    else
        return adds
    end
end