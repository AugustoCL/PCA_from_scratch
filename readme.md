# PCA from scratch with #JuliaLang
This repo contains an implementation of Principal Component Analysis (PCA) algorithm with 2 methods:
- Eigen decomposition (`:cov`) 
- SVD decomposition  (`:svd`)

The design of the function `pca` was inspired from R (prcomp) and Python (sklearn) implementations.

Besides this repo isn't a package code, this Julia code is faster than R and Python implementations with less lines of code (no C or C++ code, just pure Julia) :sunglasses: 
