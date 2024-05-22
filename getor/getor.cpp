#include <chrono>
#include <iostream>
#include <random>
#include <array>

#include <advisor-annotate.h>
#include <omp.h>

constexpr int STEP_PRINT_MOD{1};

constexpr float dt{0.1};
constexpr float T{1};
constexpr int N{static_cast<int>(T / dt)};

constexpr int N_BODYES{10000};

constexpr float MAX_POS{0.5};
constexpr float MIN_VEL{0.1};
constexpr float MAX_VEL{1};
constexpr float MIN_MASS{0.1};
constexpr float MAX_MASS{100};

constexpr float G{6.67259e-11};         // Gravitational constant
constexpr float softeningSquared{1e-6}; // Softening parameter

struct Body
{
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
    float acc_x, acc_y, acc_z;
    float mass;
};

float random_float(float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

class Simulation
{
private:
    std::array<Body, N_BODYES> particles;
    const float step_size;

public:
    Simulation(const float step_size, const std::array<Body, N_BODYES> &bodies) : particles{bodies}, step_size(step_size) {}

    void start(int n_steps)
    {
        for (int step = 0; step < n_steps; ++step)
        {
            auto start = std::chrono::high_resolution_clock::now();
            compute_func();
            update_data();
            float current_time = step * step_size;
            float total_energy = get_energy();
            auto end = std::chrono::high_resolution_clock::now();

            if (step % STEP_PRINT_MOD == 0)
            {
                std::cout << "Step: " << step << ", Time: " << current_time << ", Total Energy: " << total_energy << std::endl;
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                std::cout << "Step time is: " << duration << " microseconds" << std::endl;
            }
        }
    }

private:
#pragma omp declare simd
    void compute_func()
    {
        // Compute the acceleration for each body due to gravitational interactions
        // using Newton's law of gravitation
        ANNOTATE_SITE_BEGIN(for_compute_func);
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); ++i)
            {
                ANNOTATE_ITERATION_TASK(for_compute_func_0);
                float acc_x = 0, acc_y = 0, acc_z = 0;
                for (int j = 0; j < particles.size(); ++j)
                {
                    ANNOTATE_ITERATION_TASK(for_compute_func_0_0);
                    if (i != j)
                    {
                        float dx = particles[j].pos_x - particles[i].pos_x;
                        float dy = particles[j].pos_y - particles[i].pos_y;
                        float dz = particles[j].pos_z - particles[i].pos_z;
                        float rSquared = dx * dx + dy * dy + dz * dz + softeningSquared;
                        float inv_r = 1.0 / sqrt(rSquared);
                        float inv_r3 = inv_r * inv_r * inv_r;

                        acc_x += G * particles[j].mass * dx * inv_r3;
                        acc_y += G * particles[j].mass * dy * inv_r3;
                        acc_z += G * particles[j].mass * dz * inv_r3;
                    }
                }
                particles[i].acc_x = acc_x;
                particles[i].acc_y = acc_y;
                particles[i].acc_z = acc_z;
            }
        }
    }
#pragma omp declare simd
    void update_data()
    {
        // Use the Midpoint method to update positions and velocities
        ANNOTATE_SITE_BEGIN(for_update_data);
        {
#pragma omp parallel for simd
            for (auto &particle : particles)
            {
                ANNOTATE_ITERATION_TASK(for_update_data_0);
                // Prediction step
                float pred_vel_x = particle.vel_x + particle.acc_x * step_size / 2;
                float pred_vel_y = particle.vel_y + particle.acc_y * step_size / 2;
                float pred_vel_z = particle.vel_z + particle.acc_z * step_size / 2;

                float pred_pos_x = particle.pos_x + pred_vel_x * step_size / 2;
                float pred_pos_y = particle.pos_y + pred_vel_y * step_size / 2;
                float pred_pos_z = particle.pos_z + pred_vel_z * step_size / 2;

                // Correction step
                float avg_acc_x = particle.acc_x + particle.acc_x * step_size / 2;
                float avg_acc_y = particle.acc_y + particle.acc_y * step_size / 2;
                float avg_acc_z = particle.acc_z + particle.acc_z * step_size / 2;

                particle.vel_x += avg_acc_x * step_size;
                particle.vel_y += avg_acc_y * step_size;
                particle.vel_z += avg_acc_z * step_size;

                particle.pos_x += pred_vel_x * step_size;
                particle.pos_y += pred_vel_y * step_size;
                particle.pos_z += pred_vel_z * step_size;
            }
        }
    }
#pragma omp declare simd
    float get_energy()
    {
        // Calculate the total energy of the system
        float total_energy = 0;
#pragma omp parallel for reduction(+ : total_energy)
        for (int i = 0; i < particles.size(); ++i)
        {
            const auto &particle = particles[i];
            float kinetic_energy = 0.5 * particle.mass * (particle.vel_x * particle.vel_x + particle.vel_y * particle.vel_y + particle.vel_z * particle.vel_z);
            float potential_energy = 0;
            for (int j = 0; j < particles.size(); ++j)
            {
                if (i != j)
                {
                    const auto &other = particles[j];
                    float dx = other.pos_x - particle.pos_x;
                    float dy = other.pos_y - particle.pos_y;
                    float dz = other.pos_z - particle.pos_z;
                    float rSquared = dx * dx + dy * dy + dz * dz + softeningSquared;
                    potential_energy -= G * particle.mass * other.mass / sqrt(rSquared);
                }
            }
            total_energy += kinetic_energy + potential_energy;
        }
        return total_energy;
    }
};

int main()
{

    static std::array<Body, N_BODYES> bodies;

    for (int i = 0; i < N_BODYES; ++i)
    {
        float pos_x{random_float(-MAX_POS, MAX_POS)}, pos_y{random_float(-MAX_POS, MAX_POS)}, pos_z{random_float(-MAX_POS, MAX_POS)};
        float vel_x{random_float(MIN_VEL, MAX_VEL)}, vel_y{random_float(MIN_VEL, MAX_VEL)}, vel_z{random_float(MIN_VEL, MAX_VEL)};
        float acc_x{0}, acc_y{0}, acc_z{0};
        float mass{random_float(MIN_MASS, MAX_MASS)};

        bodies[i] = Body{pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, mass};
    }

    // Initialize simulation with a step size
    static Simulation sim(dt, bodies); // Step size

    auto start = std::chrono::high_resolution_clock::now();
    sim.start(N);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken by function: " << duration << " microseconds" << std::endl;

    return 0;
}
