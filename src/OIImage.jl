module OIImage

using OITOOLS
using LogDensityProblems
using AdvancedHMC
using Optim
using OptimPackNextGen
using Statistics
using LinearAlgebra
using Random

# SparseArrays is loaded by OITOOLS but cannot be declared as a direct dep in
# Julia 1.12 (stdlib registration restriction). Expose a helper called at runtime
# (after OITOOLS has loaded SparseArrays) to create empty correlation matrices.
"""
    spzeros_empty() -> SparseMatrixCSC{Float64,Int64}

Return an empty 0×0 sparse matrix for OIdata correlation-matrix fields.
Called at test/user code runtime after OITOOLS has loaded SparseArrays.
"""
function spzeros_empty()
    SA = Base.loaded_modules[
        Base.PkgId(Base.UUID("2f01184e-e22b-5df5-ae63-d93ebab69eaf"), "SparseArrays")]
    return SA.spzeros(Float64, 0, 0)
end

# Re-export key OITOOLS symbols so users can access them from OIImage
import OITOOLS: OIdata, setup_nfft, setup_dft, chi2_f, chi2_fg, crit_fg, regularization, reconstruct

export OIdata, setup_nfft, setup_dft, chi2_f, chi2_fg, crit_fg, regularization, reconstruct

include("new_regularizers.jl")
include("posterior.jl")
include("sa_init.jl")
include("reconstruct_map.jl")
include("reconstruct_hmc.jl")

export laplacian, good_roughness, l1l2_wavelet, extended_regularization
export OIPosterior
export sa_init, map_reconstruct, hmc_reconstruct, compare_regularizers
export spzeros_empty

end # module OIImage
