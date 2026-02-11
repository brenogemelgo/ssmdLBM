/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
    Global compile-time geometry, math, relaxation, and utility functions shared across the solver

Namespace
    block
    physics
    geometry
    relaxation
    LBM
    math
    size

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef GLOBALFUNCTIONS_CUH
#define GLOBALFUNCTIONS_CUH

#include "constants.cuh"

namespace block
{
    __device__ __host__ [[nodiscard]] static inline consteval unsigned num_block_x() noexcept
    {
        return (mesh::nx + block::nx - 1) / block::nx;
    }

    __device__ __host__ [[nodiscard]] static inline consteval unsigned num_block_y() noexcept
    {
        return (mesh::ny + block::ny - 1) / block::ny;
    }

    __device__ __host__ [[nodiscard]] static inline consteval unsigned size() noexcept
    {
        return block::nx * block::ny * block::nz;
    }
}

namespace size
{
    __device__ __host__ [[nodiscard]] static inline consteval label_t stride() noexcept
    {
        return mesh::nx * mesh::ny;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t cells() noexcept
    {
        return mesh::nx * mesh::ny * mesh::nz;
    }
}

namespace geometry
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t R2_water() noexcept
    {
        return static_cast<scalar_t>(mesh::radius_water * mesh::radius_water);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t R2_oil() noexcept
    {
        return static_cast<scalar_t>(mesh::radius_oil * mesh::radius_oil);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_x() noexcept
    {
        return static_cast<scalar_t>(mesh::nx - 1) * static_cast<scalar_t>(0.5);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_y() noexcept
    {
        return static_cast<scalar_t>(mesh::ny - 1) * static_cast<scalar_t>(0.5);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t center_z() noexcept
    {
        return static_cast<scalar_t>(mesh::nz - 1) * static_cast<scalar_t>(0.5);
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t y_pos() noexcept
    {
        return static_cast<scalar_t>(0.5) * center_y();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t z_pos() noexcept
    {
        return static_cast<scalar_t>(0.7) * center_z();
    }
}

namespace math
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t two_pi() noexcept
    {
        return static_cast<scalar_t>(2) * static_cast<scalar_t>(CUDART_PI_F);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t sqrt(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t log(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t min(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fminf(a, b);
        }
        else
        {
            return ::fmin(a, b);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t max(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaxf(a, b);
        }
        else
        {
            return ::fmax(a, b);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t cos(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tanh(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __device__ __host__ static inline void sincos(
        const scalar_t x,
        scalar_t *s,
        scalar_t *c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            ::sincosf(x, s, c);
        }
        else
        {
            ::sincos(x, s, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t fma(
        const scalar_t a,
        const scalar_t b,
        const scalar_t c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaf(a, b, c);
        }
        else
        {
            return ::fma(a, b, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t abs(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fabsf(x);
        }
        else
        {
            return ::fabs(x);
        }
    }
}

namespace relaxation
{
    __host__ __device__ [[nodiscard]] static inline consteval scalar_t visc_water() noexcept
    {
        return static_cast<scalar_t>((static_cast<double>(physics::u_water) * static_cast<double>(mesh::diam_water)) / static_cast<double>(physics::reynolds_water));
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t visc_oil() noexcept
    {
        return static_cast<scalar_t>((static_cast<double>(physics::u_oil) * static_cast<double>(mesh::diam_oil)) / static_cast<double>(physics::reynolds_oil));
    }

    __host__ __device__ [[nodiscard]] static inline constexpr scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / (static_cast<double>(0.5) + static_cast<double>(lbm::velocitySet::as2()) * static_cast<double>(nu)));
    }

    __host__ __device__ [[nodiscard]] static inline constexpr scalar_t tau_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(0.5) + static_cast<scalar_t>(lbm::velocitySet::as2()) * nu;
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_water() noexcept
    {
        return omega_from_nu(visc_water());
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_oil() noexcept
    {
        return omega_from_nu(visc_oil());
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omega_ref() noexcept
    {
        return omega_water();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_water() noexcept
    {
        return static_cast<scalar_t>(1) - omega_water();
    }

    __host__ __device__ [[nodiscard]] static inline consteval scalar_t omco_oil() noexcept
    {
        return static_cast<scalar_t>(1) - omega_oil();
    }
}

#endif