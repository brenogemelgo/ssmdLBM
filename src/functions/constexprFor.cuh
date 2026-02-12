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
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

Description
    Compile-time unrolled device loop via constexpr recursion

SourceFiles
    constexprFor.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTEXPRFOR_CUH
#define CONSTEXPRFOR_CUH

template <typename T, T v>
struct IntegralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = IntegralConstant;

    __device__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};

namespace device
{
    template <const label_t Start, const label_t End, typename F>
    __device__ inline constexpr void constexpr_for(F &&f) noexcept
    {
        if constexpr (Start < End)
        {
            f(IntegralConstant<label_t, Start>());
            if constexpr (Start + 1 < End)
            {
                device::constexpr_for<Start + 1, End>(std::forward<F>(f));
            }
        }
    }

    template <const label_t Start, const label_t End, const label_t Step, typename F>
    __device__ inline constexpr void step_constexpr_for(F &&f) noexcept
    {
        if constexpr ((Step > 0 && Start < End) || (Step < 0 && Start > End))
        {
            f(IntegralConstant<label_t, Start>());
            constexpr label_t Next = Start + Step;
            if constexpr ((Step > 0 && Next < End) || (Step < 0 && Next > End))
            {
                device::step_constexpr_for<Next, End, Step>(std::forward<F>(f));
            }
        }
    }
}

#endif