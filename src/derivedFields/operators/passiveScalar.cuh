/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Description
    Passive scalar advection–diffusion kernel (non-LBM), conservative TVD flux
    + explicit diffusion. Requires double buffer (d.c2) or packed 2*cells.

Namespace
    lbm
    derived
    passive

SourceFiles
    passiveScalar.cuh

\*---------------------------------------------------------------------------*/

#ifndef PASSIVESCALAR_CUH
#define PASSIVESCALAR_CUH

#include "fileIO/fields.cuh"

#if PASSIVE_SCALAR

// ------------------------ user-tunable knobs ------------------------
// If you don't have d.c2 in LBMFields, set this to 1 AND allocate d.c as 2*cells.
// src = d.c + 0*cells, dst = d.c + 1*cells, and host must swap which half is "current"
// OR keep "current" always in first half and swap pointers externally.
#ifndef PASSIVE_SCALAR_PACKED_DOUBLE_BUFFER
#define PASSIVE_SCALAR_PACKED_DOUBLE_BUFFER 0
#endif

// Molecular diffusivity in lattice units (dx=dt=1). Explicit diffusion stability (3D):
// kappa_eff <= ~1/6 for strict stability of forward Euler with 7-pt Laplacian.
#ifndef PASSIVE_SCALAR_KAPPA_MOL
#define PASSIVE_SCALAR_KAPPA_MOL 0.0
#endif

// Optional SGS/turbulent diffusivity (Smagorinsky-like from resolved strain rate)
// Helps “look reasonable” on coarse turbulent meshes by modeling unresolved mixing,
// but can smear sharp blobs. Keep off if you want crisp advection.
#ifndef PASSIVE_SCALAR_SGS
#define PASSIVE_SCALAR_SGS 0
#endif

#ifndef PASSIVE_SCALAR_CS
#define PASSIVE_SCALAR_CS 0.16
#endif

#ifndef PASSIVE_SCALAR_SC_T
#define PASSIVE_SCALAR_SC_T 0.70
#endif

// Clamp to local neighbor bounds to prevent overshoots (useful with limiters + diffusion).
#ifndef PASSIVE_SCALAR_CLAMP
#define PASSIVE_SCALAR_CLAMP 1
#endif

// -------------------------------------------------------------------

namespace lbm
{
    namespace passive_detail
    {
        template <class Fields>
        static constexpr bool has_c2_v = requires(Fields t) { t.c2; };

        template <class Fields>
        static constexpr bool has_u_soa3_v = requires(Fields t) { t.u; };

        template <class Fields>
        static constexpr bool has_uxuyuz_v =
            requires(Fields t) { t.ux; t.uy; t.uz; };

        __device__ __forceinline__ static scalar_t clamp_scalar(
            const scalar_t x, const scalar_t lo, const scalar_t hi) noexcept
        {
            return fmin(fmax(x, lo), hi);
        }

        // Minmod limiter (very robust, more diffusive)
        __device__ __forceinline__ static scalar_t minmod(
            const scalar_t a, const scalar_t b) noexcept
        {
            if (a * b <= static_cast<scalar_t>(0))
                return static_cast<scalar_t>(0);
            return (fabs(a) < fabs(b)) ? a : b;
        }

        // Van Leer limiter (less diffusive than minmod, still TVD)
        __device__ __forceinline__ static scalar_t vanleer(
            const scalar_t a, const scalar_t b) noexcept
        {
            if (a * b <= static_cast<scalar_t>(0))
                return static_cast<scalar_t>(0);
            const scalar_t ab = a * b;
            return static_cast<scalar_t>(2) * ab / (a + b);
        }

        __device__ __forceinline__ static scalar_t limiter(
            const scalar_t a, const scalar_t b) noexcept
        {
            // Switch here if you want: minmod(a,b) (safer) vs vanleer(a,b) (sharper)
            return vanleer(a, b);
        }

        template <class Fields>
        __device__ __forceinline__ static scalar_t ux(const Fields &d, const label_t idx) noexcept
        {
            if constexpr (has_u_soa3_v<Fields>)
            {
                return d.u[0 * size::cells() + idx];
            }
            else if constexpr (has_uxuyuz_v<Fields>)
            {
                return d.ux[idx];
            }
            else
            {
                // hard error if neither layout exists
                asm volatile("trap;");
                return static_cast<scalar_t>(0);
            }
        }

        template <class Fields>
        __device__ __forceinline__ static scalar_t uy(const Fields &d, const label_t idx) noexcept
        {
            if constexpr (has_u_soa3_v<Fields>)
            {
                return d.u[1 * size::cells() + idx];
            }
            else if constexpr (has_uxuyuz_v<Fields>)
            {
                return d.uy[idx];
            }
            else
            {
                asm volatile("trap;");
                return static_cast<scalar_t>(0);
            }
        }

        template <class Fields>
        __device__ __forceinline__ static scalar_t uz(const Fields &d, const label_t idx) noexcept
        {
            if constexpr (has_u_soa3_v<Fields>)
            {
                return d.u[2 * size::cells() + idx];
            }
            else if constexpr (has_uxuyuz_v<Fields>)
            {
                return d.uz[idx];
            }
            else
            {
                asm volatile("trap;");
                return static_cast<scalar_t>(0);
            }
        }

#if PASSIVE_SCALAR_SGS
        template <class Fields>
        __device__ __forceinline__ static scalar_t strain_mag(const Fields &d,
                                                              const label_t idx,
                                                              const label_t sx,
                                                              const label_t sy,
                                                              const label_t sz) noexcept
        {
            // Central differences for velocity gradients, dx=1
            const scalar_t uxp = ux(d, idx + sx);
            const scalar_t uxm = ux(d, idx - sx);
            const scalar_t uyp = ux(d, idx + sy);
            const scalar_t uym = ux(d, idx - sy);
            const scalar_t uzp = ux(d, idx + sz);
            const scalar_t uzm = ux(d, idx - sz);

            const scalar_t vxp = uy(d, idx + sx);
            const scalar_t vxm = uy(d, idx - sx);
            const scalar_t vyp = uy(d, idx + sy);
            const scalar_t vym = uy(d, idx - sy);
            const scalar_t vzp = uy(d, idx + sz);
            const scalar_t vzm = uy(d, idx - sz);

            const scalar_t wxp = uz(d, idx + sx);
            const scalar_t wxm = uz(d, idx - sx);
            const scalar_t wyp = uz(d, idx + sy);
            const scalar_t wym = uz(d, idx - sy);
            const scalar_t wzp = uz(d, idx + sz);
            const scalar_t wzm = uz(d, idx - sz);

            const scalar_t du_dx = static_cast<scalar_t>(0.5) * (uxp - uxm);
            const scalar_t du_dy = static_cast<scalar_t>(0.5) * (uyp - uym);
            const scalar_t du_dz = static_cast<scalar_t>(0.5) * (uzp - uzm);

            const scalar_t dv_dx = static_cast<scalar_t>(0.5) * (vxp - vxm);
            const scalar_t dv_dy = static_cast<scalar_t>(0.5) * (vyp - vym);
            const scalar_t dv_dz = static_cast<scalar_t>(0.5) * (vzp - vzm);

            const scalar_t dw_dx = static_cast<scalar_t>(0.5) * (wxp - wxm);
            const scalar_t dw_dy = static_cast<scalar_t>(0.5) * (wyp - wym);
            const scalar_t dw_dz = static_cast<scalar_t>(0.5) * (wzp - wzm);

            // Symmetric strain-rate tensor S_ij = 0.5(du_i/dx_j + du_j/dx_i)
            const scalar_t Sxx = du_dx;
            const scalar_t Syy = dv_dy;
            const scalar_t Szz = dw_dz;

            const scalar_t Sxy = static_cast<scalar_t>(0.5) * (du_dy + dv_dx);
            const scalar_t Sxz = static_cast<scalar_t>(0.5) * (du_dz + dw_dx);
            const scalar_t Syz = static_cast<scalar_t>(0.5) * (dv_dz + dw_dy);

            // |S| = sqrt(2 S_ij S_ij)
            const scalar_t s2 =
                static_cast<scalar_t>(2) * (Sxx * Sxx + Syy * Syy + Szz * Szz +
                                            static_cast<scalar_t>(2) * (Sxy * Sxy + Sxz * Sxz + Syz * Syz));
            return sqrt(fmax(s2, static_cast<scalar_t>(0)));
        }
#endif
    }

    __global__ void advectDiffuse(LBMFields d)
    {
        // ---- requirements ----
        // Need velocity field either as d.u (SoA, 3*cells) or d.ux/d.uy/d.uz
        static_assert(passive_detail::has_u_soa3_v<LBMFields> ||
                          passive_detail::has_uxuyuz_v<LBMFields>,
                      "PASSIVE_SCALAR: LBMFields must expose velocity as d.u (3*SoA) or d.ux/d.uy/d.uz.");

        // Need double buffer for c (recommended: add d.c2 to LBMFields).
        static_assert(passive_detail::has_c2_v<LBMFields> || (PASSIVE_SCALAR_PACKED_DOUBLE_BUFFER == 1),
                      "PASSIVE_SCALAR: need d.c2 in LBMFields, or set PASSIVE_SCALAR_PACKED_DOUBLE_BUFFER=1 and allocate d.c as 2*cells.");

        const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
        const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
        const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

        // 1-ghost interior (typical). Adjust if your mesh has thicker halos.
        if (x <= 0 || x >= mesh::nx - 1 ||
            y <= 0 || y >= mesh::ny - 1 ||
            z <= 0 || z >= mesh::nz - 1)
        {
            return;
        }

        const label_t idx = device::global3(x, y, z);

        // Assume x-fastest linearization: idx +/- 1, +/- nx, +/- nx*ny.
        // This matches device::global3 in typical layouts.
        const label_t sx = static_cast<label_t>(1);
        const label_t sy = static_cast<label_t>(mesh::nx);
        const label_t sz = static_cast<label_t>(mesh::nx * mesh::ny);

        // src/dst concentration buffers
        const scalar_t *c_src = nullptr;
        scalar_t *c_dst = nullptr;

        if constexpr (passive_detail::has_c2_v<LBMFields>)
        {
            c_src = d.c;
            c_dst = d.c2;
        }
        else
        {
            // Packed double buffer: user must manage which half is current outside (recommended: swap pointers).
            const label_t cells = static_cast<label_t>(size::cells());
            c_src = d.c + 0 * cells;
            c_dst = d.c + 1 * cells;
        }

        const scalar_t c0 = c_src[idx];

        // Neighbor samples for diffusion and clamps
        const scalar_t cxp = c_src[idx + sx];
        const scalar_t cxm = c_src[idx - sx];
        const scalar_t cyp = c_src[idx + sy];
        const scalar_t cym = c_src[idx - sy];
        const scalar_t czp = c_src[idx + sz];
        const scalar_t czm = c_src[idx - sz];

        // Face velocities (cell-centered average), dx=dt=1
        const scalar_t ux0 = passive_detail::ux(d, idx);
        const scalar_t uy0 = passive_detail::uy(d, idx);
        const scalar_t uz0 = passive_detail::uz(d, idx);

        const scalar_t ux_p = passive_detail::ux(d, idx + sx);
        const scalar_t ux_m = passive_detail::ux(d, idx - sx);
        const scalar_t uy_p = passive_detail::uy(d, idx + sy);
        const scalar_t uy_m = passive_detail::uy(d, idx - sy);
        const scalar_t uz_p = passive_detail::uz(d, idx + sz);
        const scalar_t uz_m = passive_detail::uz(d, idx - sz);

        const scalar_t ufx_p = static_cast<scalar_t>(0.5) * (ux0 + ux_p); // i+1/2
        const scalar_t ufx_m = static_cast<scalar_t>(0.5) * (ux_m + ux0); // i-1/2
        const scalar_t ufy_p = static_cast<scalar_t>(0.5) * (uy0 + uy_p);
        const scalar_t ufy_m = static_cast<scalar_t>(0.5) * (uy_m + uy0);
        const scalar_t ufz_p = static_cast<scalar_t>(0.5) * (uz0 + uz_p);
        const scalar_t ufz_m = static_cast<scalar_t>(0.5) * (uz_m + uz0);

        // --- TVD face reconstruction with minimal extra halo needs ---
        // We do MUSCL from the upwind cell. For negative face velocity, we need i+2.
        // If you only have 1 ghost layer, i+2 is available for interior x<=nx-3 etc.
        auto face_upwind = [&](const scalar_t uf,
                               const scalar_t cim1, const scalar_t ci, const scalar_t cip1, const scalar_t cip2,
                               const bool has_cip2) __device__ -> scalar_t
        {
            if (uf >= static_cast<scalar_t>(0))
            {
                // upwind cell is i: need (ci-cim1, cip1-ci)
                const scalar_t s = passive_detail::limiter(ci - cim1, cip1 - ci);
                return ci + static_cast<scalar_t>(0.5) * s;
            }
            else
            {
                // upwind cell is i+1: needs (cip2-cip1, cip1-ci)
                if (has_cip2)
                {
                    const scalar_t s = passive_detail::limiter(cip2 - cip1, cip1 - ci);
                    return cip1 - static_cast<scalar_t>(0.5) * s;
                }
                // fallback to 1st-order upwind if i+2 not available
                return cip1;
            }
        };

        // X fluxes
        scalar_t Fx_p, Fx_m;
        {
            // i+1/2 uses (i-1,i,i+1,i+2) if available
            const bool has_xp2 = (x <= mesh::nx - 3); // ensures x+2 <= nx-1
            const scalar_t cxp2 = has_xp2 ? c_src[idx + 2 * sx] : cxp;

            const scalar_t c_face_p = face_upwind(ufx_p, cxm, c0, cxp, cxp2, has_xp2);
            Fx_p = ufx_p * c_face_p;

            // i-1/2: shift stencil left
            const bool has_xm2 = (x >= 2);
            const scalar_t cxm2 = has_xm2 ? c_src[idx - 2 * sx] : cxm;

            // For i-1/2, "ci" is c_{i-1}, "cip1" is c_i, "cim1" is c_{i-2}, "cip2" is c_{i+1}
            // Note: negative velocity case would need c_{i+1} which we have (cxp)
            const scalar_t c_face_m = face_upwind(ufx_m, cxm2, cxm, c0, cxp, true);
            Fx_m = ufx_m * c_face_m;
        }

        // Y fluxes
        scalar_t Fy_p, Fy_m;
        {
            const bool has_yp2 = (y <= mesh::ny - 3);
            const scalar_t cyp2 = has_yp2 ? c_src[idx + 2 * sy] : cyp;

            const scalar_t c_face_p = face_upwind(ufy_p, cym, c0, cyp, cyp2, has_yp2);
            Fy_p = ufy_p * c_face_p;

            const bool has_ym2 = (y >= 2);
            const scalar_t cym2 = has_ym2 ? c_src[idx - 2 * sy] : cym;

            const scalar_t c_face_m = face_upwind(ufy_m, cym2, cym, c0, cyp, true);
            Fy_m = ufy_m * c_face_m;
        }

        // Z fluxes
        scalar_t Fz_p, Fz_m;
        {
            const bool has_zp2 = (z <= mesh::nz - 3);
            const scalar_t czp2 = has_zp2 ? c_src[idx + 2 * sz] : czp;

            const scalar_t c_face_p = face_upwind(ufz_p, czm, c0, czp, czp2, has_zp2);
            Fz_p = ufz_p * c_face_p;

            const bool has_zm2 = (z >= 2);
            const scalar_t czm2 = has_zm2 ? c_src[idx - 2 * sz] : czm;

            const scalar_t c_face_m = face_upwind(ufz_m, czm2, czm, c0, czp, true);
            Fz_m = ufz_m * c_face_m;
        }

        // Conservative divergence (dx=1)
        const scalar_t adv = -((Fx_p - Fx_m) + (Fy_p - Fy_m) + (Fz_p - Fz_m));

        // Diffusion (7-pt Laplacian), dx=1
        scalar_t kappa = static_cast<scalar_t>(PASSIVE_SCALAR_KAPPA_MOL);

#if PASSIVE_SCALAR_SGS
        {
            const scalar_t Smag = passive_detail::strain_mag(d, idx, sx, sy, sz);
            const scalar_t Cs = static_cast<scalar_t>(PASSIVE_SCALAR_CS);
            const scalar_t ScT = static_cast<scalar_t>(PASSIVE_SCALAR_SC_T);
            const scalar_t kappa_t = (Cs * Cs) * Smag / fmax(ScT, static_cast<scalar_t>(1e-12));
            kappa += kappa_t;

            // Optional hard cap to keep explicit diffusion sane (tune as needed)
            kappa = fmin(kappa, static_cast<scalar_t>(0.15));
        }
#endif

        const scalar_t lap = (cxp + cxm + cyp + cym + czp + czm - static_cast<scalar_t>(6) * c0);
        const scalar_t diff = kappa * lap;

        scalar_t c1 = c0 + adv + diff;

#if PASSIVE_SCALAR_CLAMP
        {
            // Clamp to local bounds (prevents new extrema). Use 6-neighbor bounds.
            scalar_t lo = c0, hi = c0;
            lo = fmin(lo, cxp);
            hi = fmax(hi, cxp);
            lo = fmin(lo, cxm);
            hi = fmax(hi, cxm);
            lo = fmin(lo, cyp);
            hi = fmax(hi, cyp);
            lo = fmin(lo, cym);
            hi = fmax(hi, cym);
            lo = fmin(lo, czp);
            hi = fmax(hi, czp);
            lo = fmin(lo, czm);
            hi = fmax(hi, czm);

            c1 = passive_detail::clamp_scalar(c1, lo, hi);
        }
#endif

        c_dst[idx] = c1;
    }
}

namespace derived
{
    namespace passive
    {
        constexpr std::array<host::FieldConfig, 1> fields{{
            {host::FieldID::C, "c", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d) noexcept
        {
            lbm::advectDiffuse<<<grid, block, dynamic, queue>>>(d);
        }

        __host__ static inline void free(LBMFields &d) noexcept
        {
            if (d.c)
            {
                cudaFree(d.c);
                d.c = nullptr;
            }
#if !PASSIVE_SCALAR_PACKED_DOUBLE_BUFFER
            // If you add d.c2 to LBMFields, also free it here.
            if constexpr (requires(LBMFields t) { t.c2; })
            {
                if (d.c2)
                {
                    cudaFree(d.c2);
                    d.c2 = nullptr;
                }
            }
#endif
        }
    }
}

#endif // PASSIVE_SCALAR
#endif // PASSIVESCALAR_CUH
