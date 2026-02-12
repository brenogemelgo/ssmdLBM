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
    High-order phase-field kernels computing order parameter, interface normals, curvature, and surface-tension forcing using D3Q19/D3Q27 stencils

Namespace
    phase

SourceFiles
    phaseField.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEFIELD_CUH
#define PHASEFIELD_CUH

namespace phase
{
    __global__ void computePhase(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t phi = static_cast<scalar_t>(0);
        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                phi += d.g[Q * size::cells() + idx3];
            });

        d.phi[idx3] = phi;
    }

    __global__ void computeForces(LBMFields d)
    {
        constexpr label_t sxPhi = block::nx + 4;
        constexpr label_t syPhi = block::ny + 4;
        constexpr label_t szPhi = block::nz + 4;

        constexpr label_t sxN = block::nx + 2;
        constexpr label_t syN = block::ny + 2;
        constexpr label_t szN = block::nz + 2;

        constexpr label_t phiCells = sxPhi * syPhi * szPhi;
        constexpr label_t nCells = sxN * syN * szN;
        constexpr label_t nThreads = block::nx * block::ny * block::nz;

        __shared__ scalar_t shared_phi[phiCells];
        __shared__ scalar_t shared_normx[nCells];
        __shared__ scalar_t shared_normy[nCells];
        __shared__ scalar_t shared_normz[nCells];
        __shared__ scalar_t shared_ind[nCells];

        const label_t x0 = blockIdx.x * block::nx;
        const label_t y0 = blockIdx.y * block::ny;
        const label_t z0 = blockIdx.z * block::nz;

        const label_t x = x0 + threadIdx.x;
        const label_t y = y0 + threadIdx.y;
        const label_t z = z0 + threadIdx.z;

        const bool inDomain = (x < mesh::nx && y < mesh::ny && z < mesh::nz);

        const bool active = inDomain &&
                            (x > 0 && x < mesh::nx - 1) &&
                            (y > 0 && y < mesh::ny - 1) &&
                            (z > 0 && z < mesh::nz - 1);

        const label_t tid = threadIdx.x + block::nx * (threadIdx.y + block::ny * threadIdx.z);

        const auto sidxPhi = [] __device__(label_t lx, label_t ly, label_t lz) -> label_t
        {
            return (lz * syPhi + ly) * sxPhi + lx;
        };

        const auto sidxN = [] __device__(label_t lx, label_t ly, label_t lz) -> label_t
        {
            return (lz * syN + ly) * sxN + lx;
        };

        const bool blockPhiNoBorder =
            (x0 >= 2) && (y0 >= 2) && (z0 >= 2) &&
            (x0 + block::nx <= mesh::nx - 2) &&
            (y0 + block::ny <= mesh::ny - 2) &&
            (z0 + block::nz <= mesh::nz - 2);

        for (label_t k = tid; k < phiCells; k += nThreads)
        {
            const label_t t = k / sxPhi;
            const label_t lx = k % sxPhi;
            const label_t ly = t % syPhi;
            const label_t lz = t / syPhi;

            const int gx = static_cast<int>(x0) + static_cast<int>(lx) - 2;
            const int gy = static_cast<int>(y0) + static_cast<int>(ly) - 2;
            const int gz = static_cast<int>(z0) + static_cast<int>(lz) - 2;

            if (blockPhiNoBorder)
            {
                shared_phi[k] = d.phi[device::global3(gx, gy, gz)];

                continue;
            }

            const bool in =
                (gx >= 0 && gx < mesh::nx) &&
                (gy >= 0 && gy < mesh::ny) &&
                (gz >= 0 && gz < mesh::nz);

            if (!in)
            {
                shared_phi[k] = static_cast<scalar_t>(0);

                continue;
            }

            const bool onBorder =
                (gx == 0 || gx == mesh::nx - 1) ||
                (gy == 0 || gy == mesh::ny - 1) ||
                (gz == 0 || gz == mesh::nz - 1);

            shared_phi[k] = onBorder ? static_cast<scalar_t>(0) : d.phi[device::global3(gx, gy, gz)];
        }

        __syncthreads();

        for (label_t k = tid; k < nCells; k += nThreads)
        {
            const label_t t = k / sxN;
            const label_t lx = k % sxN;
            const label_t ly = t % syN;
            const label_t lz = t / syN;

            const label_t cx = lx + 1;
            const label_t cy = ly + 1;
            const label_t cz = lz + 1;

            const scalar_t phi_xp1_yp1_z = shared_phi[sidxPhi(cx + 1, cy + 1, cz)];
            const scalar_t phi_xp1_y_zp1 = shared_phi[sidxPhi(cx + 1, cy, cz + 1)];
            const scalar_t phi_xp1_ym1_z = shared_phi[sidxPhi(cx + 1, cy - 1, cz)];
            const scalar_t phi_xp1_y_zm1 = shared_phi[sidxPhi(cx + 1, cy, cz - 1)];
            const scalar_t phi_xm1_ym1_z = shared_phi[sidxPhi(cx - 1, cy - 1, cz)];
            const scalar_t phi_xm1_y_zm1 = shared_phi[sidxPhi(cx - 1, cy, cz - 1)];
            const scalar_t phi_xm1_yp1_z = shared_phi[sidxPhi(cx - 1, cy + 1, cz)];
            const scalar_t phi_xm1_y_zp1 = shared_phi[sidxPhi(cx - 1, cy, cz + 1)];
            const scalar_t phi_x_yp1_zp1 = shared_phi[sidxPhi(cx, cy + 1, cz + 1)];
            const scalar_t phi_x_yp1_zm1 = shared_phi[sidxPhi(cx, cy + 1, cz - 1)];
            const scalar_t phi_x_ym1_zm1 = shared_phi[sidxPhi(cx, cy - 1, cz - 1)];
            const scalar_t phi_x_ym1_zp1 = shared_phi[sidxPhi(cx, cy - 1, cz + 1)];

            scalar_t sgx = lbm::velocitySet::w_1() * (shared_phi[sidxPhi(cx + 1, cy, cz)] - shared_phi[sidxPhi(cx - 1, cy, cz)]) +
                           lbm::velocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                      phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                      phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                      phi_xp1_y_zm1 - phi_xm1_y_zp1);

            scalar_t sgy = lbm::velocitySet::w_1() * (shared_phi[sidxPhi(cx, cy + 1, cz)] - shared_phi[sidxPhi(cx, cy - 1, cz)]) +
                           lbm::velocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                      phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                      phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                      phi_x_yp1_zm1 - phi_x_ym1_zp1);

            scalar_t sgz = lbm::velocitySet::w_1() * (shared_phi[sidxPhi(cx, cy, cz + 1)] - shared_phi[sidxPhi(cx, cy, cz - 1)]) +
                           lbm::velocitySet::w_2() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                      phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                      phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                      phi_x_ym1_zp1 - phi_x_yp1_zm1);

            if constexpr (lbm::velocitySet::Q() == 27)
            {
                const scalar_t phi_xp1_yp1_zp1 = shared_phi[sidxPhi(cx + 1, cy + 1, cz + 1)];
                const scalar_t phi_xp1_yp1_zm1 = shared_phi[sidxPhi(cx + 1, cy + 1, cz - 1)];
                const scalar_t phi_xp1_ym1_zp1 = shared_phi[sidxPhi(cx + 1, cy - 1, cz + 1)];
                const scalar_t phi_xp1_ym1_zm1 = shared_phi[sidxPhi(cx + 1, cy - 1, cz - 1)];
                const scalar_t phi_xm1_ym1_zm1 = shared_phi[sidxPhi(cx - 1, cy - 1, cz - 1)];
                const scalar_t phi_xm1_ym1_zp1 = shared_phi[sidxPhi(cx - 1, cy - 1, cz + 1)];
                const scalar_t phi_xm1_yp1_zm1 = shared_phi[sidxPhi(cx - 1, cy + 1, cz - 1)];
                const scalar_t phi_xm1_yp1_zp1 = shared_phi[sidxPhi(cx - 1, cy + 1, cz + 1)];

                sgx += lbm::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                            phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                            phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                            phi_xp1_ym1_zm1 - phi_xm1_yp1_zp1);

                sgy += lbm::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                            phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                            phi_xm1_yp1_zm1 - phi_xp1_ym1_zp1 +
                                            phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);

                sgz += lbm::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                            phi_xm1_ym1_zp1 - phi_xp1_yp1_zm1 +
                                            phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                            phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);
            }

            const scalar_t gx = lbm::velocitySet::as2() * sgx;
            const scalar_t gy = lbm::velocitySet::as2() * sgy;
            const scalar_t gz = lbm::velocitySet::as2() * sgz;

            const scalar_t ind = math::sqrt(gx * gx + gy * gy + gz * gz);
            const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

            shared_normx[k] = gx * invInd;
            shared_normy[k] = gy * invInd;
            shared_normz[k] = gz * invInd;
            shared_ind[k] = ind;
        }

        __syncthreads();

        if (!inDomain)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        if (!active)
        {
            d.fsx[idx3] = static_cast<scalar_t>(0);
            d.fsy[idx3] = static_cast<scalar_t>(0);
            d.fsz[idx3] = static_cast<scalar_t>(0);
            return;
        }

        const label_t lxN = threadIdx.x + 1;
        const label_t lyN = threadIdx.y + 1;
        const label_t lzN = threadIdx.z + 1;

        scalar_t scx = lbm::velocitySet::w_1() * (shared_normx[sidxN(lxN + 1, lyN, lzN)] - shared_normx[sidxN(lxN - 1, lyN, lzN)]) +
                       lbm::velocitySet::w_2() * (shared_normx[sidxN(lxN + 1, lyN + 1, lzN)] - shared_normx[sidxN(lxN - 1, lyN - 1, lzN)] +
                                                  shared_normx[sidxN(lxN + 1, lyN, lzN + 1)] - shared_normx[sidxN(lxN - 1, lyN, lzN - 1)] +
                                                  shared_normx[sidxN(lxN + 1, lyN - 1, lzN)] - shared_normx[sidxN(lxN - 1, lyN + 1, lzN)] +
                                                  shared_normx[sidxN(lxN + 1, lyN, lzN - 1)] - shared_normx[sidxN(lxN - 1, lyN, lzN + 1)]);

        scalar_t scy = lbm::velocitySet::w_1() * (shared_normy[sidxN(lxN, lyN + 1, lzN)] - shared_normy[sidxN(lxN, lyN - 1, lzN)]) +
                       lbm::velocitySet::w_2() * (shared_normy[sidxN(lxN + 1, lyN + 1, lzN)] - shared_normy[sidxN(lxN - 1, lyN - 1, lzN)] +
                                                  shared_normy[sidxN(lxN, lyN + 1, lzN + 1)] - shared_normy[sidxN(lxN, lyN - 1, lzN - 1)] +
                                                  shared_normy[sidxN(lxN - 1, lyN + 1, lzN)] - shared_normy[sidxN(lxN + 1, lyN - 1, lzN)] +
                                                  shared_normy[sidxN(lxN, lyN + 1, lzN - 1)] - shared_normy[sidxN(lxN, lyN - 1, lzN + 1)]);

        scalar_t scz = lbm::velocitySet::w_1() * (shared_normz[sidxN(lxN, lyN, lzN + 1)] - shared_normz[sidxN(lxN, lyN, lzN - 1)]) +
                       lbm::velocitySet::w_2() * (shared_normz[sidxN(lxN + 1, lyN, lzN + 1)] - shared_normz[sidxN(lxN - 1, lyN, lzN - 1)] +
                                                  shared_normz[sidxN(lxN, lyN + 1, lzN + 1)] - shared_normz[sidxN(lxN, lyN - 1, lzN - 1)] +
                                                  shared_normz[sidxN(lxN - 1, lyN, lzN + 1)] - shared_normz[sidxN(lxN + 1, lyN, lzN - 1)] +
                                                  shared_normz[sidxN(lxN, lyN - 1, lzN + 1)] - shared_normz[sidxN(lxN, lyN + 1, lzN - 1)]);

        if constexpr (lbm::velocitySet::Q() == 27)
        {
            scx += lbm::D3Q27::w_3() * (shared_normx[sidxN(lxN + 1, lyN + 1, lzN + 1)] - shared_normx[sidxN(lxN - 1, lyN - 1, lzN - 1)] +
                                        shared_normx[sidxN(lxN + 1, lyN + 1, lzN - 1)] - shared_normx[sidxN(lxN - 1, lyN - 1, lzN + 1)] +
                                        shared_normx[sidxN(lxN + 1, lyN - 1, lzN + 1)] - shared_normx[sidxN(lxN - 1, lyN + 1, lzN - 1)] +
                                        shared_normx[sidxN(lxN + 1, lyN - 1, lzN - 1)] - shared_normx[sidxN(lxN - 1, lyN + 1, lzN + 1)]);

            scy += lbm::D3Q27::w_3() * (shared_normy[sidxN(lxN + 1, lyN + 1, lzN + 1)] - shared_normy[sidxN(lxN - 1, lyN - 1, lzN - 1)] +
                                        shared_normy[sidxN(lxN + 1, lyN + 1, lzN - 1)] - shared_normy[sidxN(lxN - 1, lyN - 1, lzN + 1)] +
                                        shared_normy[sidxN(lxN - 1, lyN + 1, lzN - 1)] - shared_normy[sidxN(lxN + 1, lyN - 1, lzN + 1)] +
                                        shared_normy[sidxN(lxN - 1, lyN + 1, lzN + 1)] - shared_normy[sidxN(lxN + 1, lyN - 1, lzN - 1)]);

            scz += lbm::D3Q27::w_3() * (shared_normz[sidxN(lxN + 1, lyN + 1, lzN + 1)] - shared_normz[sidxN(lxN - 1, lyN - 1, lzN - 1)] +
                                        shared_normz[sidxN(lxN - 1, lyN - 1, lzN + 1)] - shared_normz[sidxN(lxN + 1, lyN + 1, lzN - 1)] +
                                        shared_normz[sidxN(lxN + 1, lyN - 1, lzN + 1)] - shared_normz[sidxN(lxN - 1, lyN + 1, lzN - 1)] +
                                        shared_normz[sidxN(lxN - 1, lyN + 1, lzN + 1)] - shared_normz[sidxN(lxN + 1, lyN - 1, lzN - 1)]);
        }

        const label_t cN = sidxN(lxN, lyN, lzN);

        const scalar_t curvature = lbm::velocitySet::as2() * (scx + scy + scz);
        const scalar_t stCurv = -physics::sigma * curvature * shared_ind[cN];

        d.fsx[idx3] = stCurv * shared_normx[cN];
        d.fsy[idx3] = stCurv * shared_normy[cN];
        d.fsz[idx3] = stCurv * shared_normz[cN];
    }
}

#endif