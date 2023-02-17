using Rimu.Hamiltonians: check_address_type, build_sparse_matrix_from_LO, LOStructure
import Rimu.Hamiltonians: _bsr_ensure_symmetry
# using SparseArrays: nnz, spzeros

function _bsr_ensure_symmetry(
    ::LOStructure, h::HarmonicOscillatorWeak, addr;
    sizelim=10^6, test_approx_symmetry=true, basis_only = false, kwargs...
)
    dimension(Float64, h) < sizelim || throw(ArgumentError("dimension larger than sizelim"))
    check_address_type(h, addr)
    if basis_only
        sm, basis = build_basis_only_from_LO(h, addr; kwargs...)
    else
        sm, basis = build_sparse_matrix_from_LO(h, addr; kwargs...)
    end  
    return BasisSetRep(sm, basis, h)
end

# calculate the basis without storing the sparse matrix
function build_basis_only_from_LO(
    ham, address=starting_address(ham);
    cutoff=nothing,
    filter=isnothing(cutoff) ? nothing : (a -> diagonal_element(ham, a) ≤ cutoff),
    nnzs=dimension(ham),
    sort=false, kwargs...,
)
    if !isnothing(filter) && !filter(address)
        throw(ArgumentError(string(
            "Starting address does not pass `filter`. ",
            "Please pick a different address or a different filter."
        )))
    end
    T = eltype(ham)
    adds = [address]          # Queue of addresses. Also returned as the basis.
    dict = Dict(address => 1) # Mapping from addresses to indices
    col = Dict{Int,T}()       # Temporary column storage
    sizehint!(col, num_offdiagonals(ham, address))

    i = 0
    while i < length(adds)
        i += 1
        add = adds[i]

        empty!(col)
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
            if !iszero(j)
                col[j] = get(col, j, zero(T)) + v
            end
        end
    end

    if sort
        perm = sortperm(adds; kwargs...)
        return nothing, permute!(adds, perm)
    else
        return nothing, adds
    end
end