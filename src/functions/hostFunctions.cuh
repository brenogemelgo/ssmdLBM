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
   Host-side I/O, diagnostics, and utility routines for simulation setup and data output

Namespace
    host

SourceFiles
    hostFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef HOSTFUNCTIONS_CUH
#define HOSTFUNCTIONS_CUH

#include "globalFunctions.cuh"

namespace host
{
    __host__ [[nodiscard]] static inline std::string createSimulationDirectory(
        const std::string &VELOCITY_SET,
        const std::string &SIM_ID)
    {
        std::filesystem::path BASE_DIR = std::filesystem::current_path();

        std::filesystem::path SIM_DIR = BASE_DIR / "bin" / VELOCITY_SET / SIM_ID;

        std::error_code EC;
        std::filesystem::create_directories(SIM_DIR, EC);

        return SIM_DIR.string() + std::string(1, std::filesystem::path::preferred_separator);
    }

    __host__ [[gnu::cold]] static inline void generateSimulationInfoFile(
        const std::string &SIM_DIR,
        const std::string &SIM_ID,
        const std::string &VELOCITY_SET)
    {
        std::filesystem::path INFO_PATH = std::filesystem::path(SIM_DIR) / (SIM_ID + "_info.txt");

        try
        {
            std::ofstream file(INFO_PATH, std::ios::out | std::ios::trunc);
            if (!file.is_open())
            {
                std::cerr << "Error opening file: " << INFO_PATH.string() << std::endl;

                return;
            }

            file << "---------------------------- SIMULATION METADATA ----------------------------\n"
                 << "ID:                 " << SIM_ID << '\n'
                 << "Velocity set:       " << VELOCITY_SET << '\n'
                 << "Water velocity:     " << physics::u_water << '\n'
                 << "Oil velocity:       " << physics::u_oil << '\n'
                 << "Water Reynolds:     " << physics::reynolds_water << '\n'
                 << "Oil Reynolds:       " << physics::reynolds_oil << '\n'
                 << "Weber number:       " << physics::weber << "\n\n"
                 << "Domain size:        NX=" << mesh::nx << ", NY=" << mesh::ny << ", NZ=" << mesh::nz << '\n'
                 << "Water diameter:     D=" << mesh::diam_water << '\n'
                 << "Oil diameter:       D=" << mesh::diam_oil << '\n'
                 << "Timesteps:          " << NSTEPS << '\n'
                 << "Output interval:    " << MACRO_SAVE << "\n\n"
                 << "-----------------------------------------------------------------------------\n";

            file.close();

            std::cout << "Simulation information file created in: " << INFO_PATH.string() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error generating information file: " << e.what() << std::endl;
        }
    }

    __host__ [[gnu::cold]] static inline void printDiagnostics(const std::string &VELOCITY_SET) noexcept
    {
        std::cout << "\n---------------------------- SIMULATION METADATA ----------------------------\n"
                  << "Velocity set:       " << VELOCITY_SET << '\n'
                  << "Water velocity:     " << physics::u_water << '\n'
                  << "Oil velocity:       " << physics::u_oil << '\n'
                  << "Water Reynolds:     " << physics::reynolds_water << '\n'
                  << "Oil Reynolds:       " << physics::reynolds_oil << '\n'
                  << "Weber number:       " << physics::weber << '\n'
                  << "NX:                 " << mesh::nx << '\n'
                  << "NY:                 " << mesh::ny << '\n'
                  << "NZ:                 " << mesh::nz << '\n'
                  << "Water diameter:     " << mesh::diam_water << '\n'
                  << "Oil diameter:       " << mesh::diam_oil << '\n'
                  << "Cells:              " << size::cells() << '\n'
                  << "-----------------------------------------------------------------------------\n\n";
    }

    template <const size_t SIZE = size::cells()>
    __host__ [[gnu::cold]] static inline void copyAndSaveToBinary(
        const scalar_t *d_data,
        const std::string &SIM_DIR,
        const label_t STEP,
        const std::string &VAR_NAME)
    {
        static thread_local std::vector<scalar_t> host_data;
        if (host_data.size() != SIZE)
        {
            host_data.resize(SIZE);
        }

        checkCudaErrors(cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(scalar_t), cudaMemcpyDeviceToHost));

        std::ostringstream STEP_SAVE;
        STEP_SAVE << std::setw(6) << std::setfill('0') << STEP;
        const std::string filename = VAR_NAME + STEP_SAVE.str() + ".bin";
        const std::filesystem::path OUT_PATH = std::filesystem::path(SIM_DIR) / filename;

        std::ofstream file(OUT_PATH, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            std::cerr << "Error opening file " << OUT_PATH.string() << " for writing." << std::endl;

            return;
        }

        file.write(reinterpret_cast<const char *>(host_data.data()), static_cast<std::streamsize>(host_data.size() * sizeof(scalar_t)));
        file.close();
    }

    __host__ [[nodiscard]] [[gnu::cold]] static inline int setDeviceFromEnv() noexcept
    {
        int dev = 0;
        if (const char *env = std::getenv("GPU_INDEX"))
        {
            char *end = nullptr;
            long v = std::strtol(env, &end, 10);

            if (end != env && v >= 0)
            {
                dev = static_cast<int>(v);
            }
        }

        cudaError_t err = cudaSetDevice(dev);
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));

            return -1;
        }

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        {
            std::printf("Using GPU %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
        }
        else
        {
            std::printf("Using GPU %d (unknown properties)\n", dev);
        }

        return dev;
    }

    __host__ [[nodiscard]] static inline constexpr unsigned divUp(
        const unsigned a,
        const unsigned b) noexcept
    {
        return (a + b - 1u) / b;
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesScalar() noexcept
    {
        return static_cast<size_t>(size::cells()) * sizeof(scalar_t);
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesF() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(lbm::velocitySet::Q()) * sizeof(pop_t);
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesG() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(phase::velocitySet::Q()) * sizeof(scalar_t);
    }
}

#endif