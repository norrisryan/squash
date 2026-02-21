module OIImage

using OITOOLS
using LogDensityProblems
using AdvancedHMC
using Pathfinder
using Statistics
using LinearAlgebra
using Random

# Re-export key OITOOLS symbols so users can access them from OIImage
import OITOOLS: OIdata, setup_nfft, setup_dft, chi2_fg, crit_fg, regularization, reconstruct

export OIdata, setup_nfft, setup_dft, chi2_fg, crit_fg, regularization, reconstruct

include("new_regularizers.jl")
include("posterior.jl")
include("reconstruct_hmc.jl")

export laplacian, good_roughness, l1l2_wavelet, extended_regularization
export OIPosterior
export hmc_reconstruct

end # module OIImage
