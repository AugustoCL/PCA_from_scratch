# load packages -----------------------------------------------------------------------------
using Statistics: mean, std, cov, svd
using LinearAlgebra: eigen, svd
using PrettyTables


# struct and functions ----------------------------------------------------------------------
struct PCAResult{T<:Real} 
    totalvar::T
    components::Matrix{T}
    λ::Vector{T}
    variance::Vector{T}
    varianceratio::Vector{T}
    method::Symbol
    k::Int
end

function pca(X::AbstractMatrix{<:Real}, maxdim::T;  method::Symbol=:cov,
             iscentered::Bool=false, isscaled::Bool=false) where {T<:Real}
    n, p = size(X)

    # check the arguments
    method ∉ (:cov, :svd) && 
        throw(ArgumentError("method=:$method is not valid. Pass :cov or :svd."))  
    p ≥ maxdim > 0 ||
        throw(ArgumentError("`maxdim` selected is not valid. " *
            "Select a `maxdim` below the dimension of X (n < dim(X)) or " *
            "the desired explained variance (0.0 < n < 1.0)"))

    # standardize the data
    μ = iscentered ? zeros(1, p) : mean(X, dims=1)
    σ = isscaled   ? ones(1, p)  : std(X, dims=1)
    Z = (X .- μ) ./ σ

    # apply the chosen PCA method (:cov or :svd)
    if method == :cov
        Σ = cov(Z) 
        λ, V = eigen(Σ, sortby=λ -> -real(λ))        
    else 
        SVD = svd(Z')
        V = SVD.U
        λ = abs2.(SVD.S) ./ (n-1)
    end

    # calculate the explained variance
    pvars = (λ ./ sum(λ))::Vector{Float64}
    cumvar = cumsum(pvars)
    
    # select the k components desired    
    k = maxdim ≥ one(T) ? round(Int, maxdim) : count(<(maxdim), cumvar) + 1

    return PCAResult(cumvar[k], V[:, 1:k], λ[1:k], pvars[1:k], cumvar[1:k], method, k)
end

pca(X::AbstractMatrix{T}; kwargs...) where {T<:Real} = pca(X, size(X, 2); kwargs...)


# Print function ----------------------------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", pc::PCAResult{T}) where {T<:Real}
    pcnms = [Symbol(""); Symbol.("PC", 1:(pc.k))]
    varnms = ["Variance (λ)", "Variance explained", "Cumulative variance"];
    vartable = [varnms [pc.λ pc.variance pc.varianceratio]'];
    pretty_table(io, pc.components, title="PCA Eigenvectors", noheader=true)
    pretty_table(io, vartable, header=pcnms, nosubheader=true, title="Variance Explained Table")
end
