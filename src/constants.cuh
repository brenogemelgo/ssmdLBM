/*---------------------------------------------------------------------------*\
|                                                                             |
| ssmdLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/ssmdLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
   Centralized compile-time configuration of velocity sets, flow cases, mesh, and physical parameters

Namespace
    lbm
    phase

SourceFiles
    constants.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "cuda/utils.cuh"
#include "structs/LBMFields.cuh"
#include "functions/constexprFor.cuh"
#include "velocitySet/VelocitySet.cuh"

namespace lbm
{
#if defined(VS_D3Q19)
    using velocitySet = D3Q19;
#elif defined(VS_D3Q27)
    using velocitySet = D3Q27;
#endif
}

namespace phase
{
    using velocitySet = lbm::D3Q7;
}

#define RUN_MODE
// #define SAMPLE_MODE
// #define PROFILE_MODE

#if defined(RUN_MODE)

static constexpr int MACRO_SAVE = 1000;
static constexpr int NSTEPS = 100000;

#elif defined(SAMPLE_MODE)

static constexpr int MACRO_SAVE = 100;
static constexpr int NSTEPS = 1000;

#elif defined(PROFILE_MODE)

static constexpr int MACRO_SAVE = 1;
static constexpr int NSTEPS = 0;

#endif

namespace mesh
{
    static constexpr label_t res = 128;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res * 2;
    static constexpr label_t nz = res * 2;

    static constexpr int diam_water = 13;
    static constexpr int diam_oil = 13;

    static constexpr int radius_water = diam_water / 2;
    static constexpr int radius_oil = diam_oil / 2;
}

namespace physics
{
    static constexpr scalar_t u_water = static_cast<scalar_t>(0.05);
    static constexpr scalar_t u_oil = static_cast<scalar_t>(0.05);

    static constexpr int reynolds_water = 1400;
    static constexpr int reynolds_oil = 450;

    static constexpr int weber = 500;
    static constexpr scalar_t sigma = (u_oil * u_oil * mesh::diam_oil) / weber;

    static constexpr scalar_t interface_width = static_cast<scalar_t>(4); // continuum interface width. discretization may change it a little

    static constexpr scalar_t tau_g = static_cast<scalar_t>(1);                                            // phase field relaxation time
    static constexpr scalar_t diff_int = phase::velocitySet::cs2() * (tau_g - static_cast<scalar_t>(0.5)); // interfacial diffusivity
    static constexpr scalar_t kappa = static_cast<scalar_t>(4) * diff_int / interface_width;               // sharpening parameter
    static constexpr scalar_t gamma = kappa / phase::velocitySet::cs2();
}

#endif