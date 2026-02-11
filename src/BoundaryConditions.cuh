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
    Unified device-side implementation of inflow, outflow, and periodic LBM boundary conditions

Namespace
    lbm

SourceFiles
    BoundaryConditions.cuh

\*---------------------------------------------------------------------------*/

#ifndef BOUNDARYCONDITIONS_CUH
#define BOUNDARYCONDITIONS_CUH

namespace lbm
{
    class BoundaryConditions
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval BoundaryConditions(){};

        __device__ static inline void applyWaterInflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || z >= mesh::nz)
            {
                return;
            }

            const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
            const scalar_t dz = static_cast<scalar_t>(z) - geometry::z_pos();
            const scalar_t r2 = dx * dx + dz * dz;

            if (r2 > geometry::R2_water())
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, 0, z);
            const label_t idx3_yp1 = device::global3(x, 1, z);

            const scalar_t rho = static_cast<scalar_t>(1);
            const scalar_t phi = static_cast<scalar_t>(0);
            const scalar_t ux = static_cast<scalar_t>(0);
            const scalar_t uy = physics::u_water;
            const scalar_t uz = static_cast<scalar_t>(0);

            d.rho[idx3_bnd] = rho;
            d.phi[idx3_bnd] = phi;
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cy<Q>() == 1)
                    {
                        const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                        const label_t zz = z + static_cast<label_t>(velocitySet::cz<Q>());

                        const label_t fluidNode = device::global3(xx, 1, zz);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_water() * fneq);
                    }
                });

            d.g[3 * size::cells() + idx3_yp1] = phase::velocitySet::w<3>() * phi * (static_cast<scalar_t>(1) + phase::velocitySet::as2() * uy);
        }

        __device__ static inline void applyOilInflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
            const scalar_t dy = static_cast<scalar_t>(y) - geometry::y_pos();
            const scalar_t r2 = dx * dx + dy * dy;

            if (r2 > geometry::R2_oil())
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, 0);
            const label_t idx3_zp1 = device::global3(x, y, 1);

            const scalar_t rho = static_cast<scalar_t>(1);
            const scalar_t phi = static_cast<scalar_t>(1);
            const scalar_t ux = static_cast<scalar_t>(0);
            const scalar_t uy = static_cast<scalar_t>(0);
            const scalar_t uz = physics::u_oil;

            d.rho[idx3_bnd] = rho;
            d.phi[idx3_bnd] = phi;
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == 1)
                    {
                        const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                        const label_t yy = y + static_cast<label_t>(velocitySet::cy<Q>());

                        const label_t fluidNode = device::global3(xx, yy, 1);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_oil() * fneq);
                    }
                });

            d.g[5 * size::cells() + idx3_zp1] = phase::velocitySet::w<5>() * phi * (static_cast<scalar_t>(1) + phase::velocitySet::as2() * uz);
        }

        __device__ static inline void applyOutflowY(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || z >= mesh::nz)
            {
                return;
            }

            if (x == 0 || x == mesh::nx - 1 || z == 0 || z == mesh::nz - 1)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, mesh::ny - 1, z);
            const label_t idx3_ym1 = device::global3(x, mesh::ny - 2, z);

            d.rho[idx3_bnd] = d.rho[idx3_ym1];
            d.phi[idx3_bnd] = d.phi[idx3_ym1];
            d.ux[idx3_bnd] = d.ux[idx3_ym1];
            d.uy[idx3_bnd] = d.uy[idx3_ym1];
            d.uz[idx3_bnd] = d.uz[idx3_ym1];

            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];
            const scalar_t ux = d.ux[idx3_bnd];
            const scalar_t uy = d.uy[idx3_bnd];
            const scalar_t uz = d.uz[idx3_bnd];

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cy<Q>() == -1)
                    {
                        const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                        const label_t zz = z + static_cast<label_t>(velocitySet::cz<Q>());

                        const label_t fluidNode = device::global3(xx, mesh::ny - 2, zz);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_ref() * fneq);
                    }
                });

            d.g[4 * size::cells() + idx3_ym1] = phase::velocitySet::w<4>() * phi * (static_cast<scalar_t>(1) - phase::velocitySet::as2() * uy);
        }

        __device__ static inline void applyOutflowZ(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            if (x == 0 || x == mesh::nx - 1 || y == 0 || y == mesh::ny - 1)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            d.rho[idx3_bnd] = d.rho[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];
            d.ux[idx3_bnd] = d.ux[idx3_zm1];
            d.uy[idx3_bnd] = d.uy[idx3_zm1];
            d.uz[idx3_bnd] = d.uz[idx3_zm1];

            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];
            const scalar_t ux = d.ux[idx3_bnd];
            const scalar_t uy = d.uy[idx3_bnd];
            const scalar_t uz = d.uz[idx3_bnd];

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == -1)
                    {
                        const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                        const label_t yy = y + static_cast<label_t>(velocitySet::cy<Q>());

                        const label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_ref() * fneq);
                    }
                });

            d.g[6 * size::cells() + idx3_zm1] = phase::velocitySet::w<6>() * phi * (static_cast<scalar_t>(1) - phase::velocitySet::as2() * uz);
        }

        __device__ static inline void periodicX(LBMFields d)
        {
            const label_t y = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (y <= 0 || y >= mesh::ny - 1 || z <= 0 || z >= mesh::nz - 1)
            {
                return;
            }

            const label_t bL = device::global3(1, y, z);
            const label_t bR = device::global3(mesh::nx - 2, y, z);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cx<Q>() > 0)
                    {
                        d.f[Q * size::cells() + bL] = d.f[Q * size::cells() + bR];
                    }
                    if constexpr (velocitySet::cx<Q>() < 0)
                    {
                        d.f[Q * size::cells() + bR] = d.f[Q * size::cells() + bL];
                    }
                });

            device::constexpr_for<0, phase::velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (phase::velocitySet::cx<Q>() > 0)
                    {
                        d.g[Q * size::cells() + bL] = d.g[Q * size::cells() + bR];
                    }
                    if constexpr (phase::velocitySet::cx<Q>() < 0)
                    {
                        d.g[Q * size::cells() + bR] = d.g[Q * size::cells() + bL];
                    }
                });

            // Copy to ghost layer (periodic wrapping)
            const label_t gL = device::global3(0, y, z);
            const label_t gR = device::global3(mesh::nx - 1, y, z);

            d.phi[gL] = d.phi[bR];
            d.phi[gR] = d.phi[bL];

            d.ux[gL] = d.ux[bR];
            d.ux[gR] = d.ux[bL];

            d.uy[gL] = d.uy[bR];
            d.uy[gR] = d.uy[bL];

            d.uz[gL] = d.uz[bR];
            d.uz[gR] = d.uz[bL];
        }

    private:
        // No private methods
    };
}

#endif
