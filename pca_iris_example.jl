# load packages, pca and data ---------------------------------------------------------------
using RDatasets, DataFrames
include("pca_functions.jl")

iris = dataset("datasets", "iris");
X = Matrix(iris[:, 1:end-1]);

# Apply PCA  --------------------------------------------------------------------------------
pca1 = pca(X, 0.98)
pca2 = pca(X, 3, method=:svd)
pca3 = pca(X)

# Benchmark ---------------------------------------------------------------------------------
using BenchmarkTools
@benchmark pca(X, 0.98)