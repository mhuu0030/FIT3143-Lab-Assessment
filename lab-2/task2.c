/**
 * FIT3143 Parallel Computing - Lab 2, Task 2
 * Hybrid Open MPI + OpenMP Prime Search (strictly less than n)
 * Instrumented for Amdahl's Law Performance Analysis (Task 3).
 *
 * Compilation:
 *   mpicc -Wall -O2 -fopenmp task2.c -o task2 -lm
 *
 * Execution:
 *   mpirun -np <num_procs> ./task2 <n> [threads_per_process]
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization and checks odd divisors only.
 */
bool is_prime(int k) {
    if (k <= 1) return false;
    if (k == 2) return true;
    if (k % 2 == 0) return false;

    int limit = (int)sqrt((double)k);
    for (int i = 3; i <= limit; i += 2) {
        if (k % i == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int provided;

    // Initialize MPI with thread support (MPI calls restricted to main thread)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    int num_threads = 4; // Default OpenMP thread count per MPI process

    // Parse command line arguments on root rank
    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: mpirun -np <procs> %s <n> [threads_per_process]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Error: n must be a positive integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (argc >= 3) {
            num_threads = atoi(argv[2]);
            if (num_threads <= 0) num_threads = 1;
        }
    }

    // Synchronize all ranks before starting total wall-clock timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_total = MPI_Wtime();

    // Broadcast input parameters to all ranks (included in total time)
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Set thread concurrency for this MPI rank
    omp_set_num_threads(num_threads);

    // Cyclic workload distribution across odd candidate integers
    int mpi_step = 2 * size;
    int mpi_first = 3 + (2 * rank);

    int local_capacity = (n > mpi_first) ? ((n - mpi_first) / mpi_step + 2) : 1;
    bool *local_flags = (bool *)calloc(local_capacity, sizeof(bool));
    int *local_primes = (int *)malloc(local_capacity * sizeof(int));

    if (local_flags == NULL || local_primes == NULL) {
        fprintf(stderr, "[Rank %d] Error: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Array to record OpenMP thread CPU times for load balance diagnostics
    double *thread_times = (double *)calloc(num_threads, sizeof(double));

    /* =========================================================
     * TASK 3: RIGOROUS PARALLEL PHASE MEASUREMENT
     * Barrier ensures all processes enter computation simultaneously.
     * ========================================================= */
    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel_phase = MPI_Wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        struct timespec t_start_thread, t_end_thread;

        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_start_thread);

        // Dynamic work-sharing schedule
        #pragma omp for schedule(dynamic, 128)
        for (int i = mpi_first; i < n; i += mpi_step) {
            if (is_prime(i)) {
                int idx = (i - mpi_first) / mpi_step;
                local_flags[idx] = true;
            }
        }

        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_end_thread);
        thread_times[tid] = (t_end_thread.tv_sec - t_start_thread.tv_sec) +
                            (t_end_thread.tv_nsec - t_start_thread.tv_nsec) / 1e9;
    }

    // Pack discovered primes into sequential local buffer
    int local_count = 0;
    for (int i = mpi_first; i < n; i += mpi_step) {
        int idx = (i - mpi_first) / mpi_step;
        if (local_flags[idx]) {
            local_primes[local_count++] = i;
        }
    }

    // Barrier ensures the parallel phase duration accounts for the slowest worker rank
    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel_phase = MPI_Wtime();
    double local_parallel_duration = end_parallel_phase - start_parallel_phase;

    free(local_flags);

    /* =========================================================
     * COMMUNICATION & GATHER PHASE (SERIAL / FABRIC OVERHEAD)
     * ========================================================= */
    int *recvcounts = NULL;
    int *displs = NULL;
    int *all_primes_gathered = NULL;
    int total_gathered_primes = 0;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        if (recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "[Rank 0] Error: Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Gather count of primes discovered by each rank
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        total_gathered_primes = recvcounts[0];
        for (int p = 1; p < size; p++) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
            total_gathered_primes += recvcounts[p];
        }

        all_primes_gathered = (int *)malloc(total_gathered_primes * sizeof(int));
        if (all_primes_gathered == NULL && total_gathered_primes > 0) {
            fprintf(stderr, "[Rank 0] Error: Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Gather prime arrays into root process
    MPI_Gatherv(local_primes, local_count, MPI_INT,
                all_primes_gathered, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    free(local_primes);

    /* =========================================================
     * ROOT SERIAL WORK (SORTING & FILE OUTPUT)
     * ========================================================= */
    if (rank == 0) {
        bool *final_map = (bool *)calloc(n, sizeof(bool));
        if (final_map == NULL) {
            fprintf(stderr, "[Rank 0] Error: Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (n > 2) final_map[2] = true;

        for (int i = 0; i < total_gathered_primes; i++) {
            int prime_val = all_primes_gathered[i];
            if (prime_val < n) {
                final_map[prime_val] = true;
            }
        }

        FILE *file = NULL;
        if (n >= 100) {
            file = fopen("task2primes.txt", "w");
            if (file == NULL) {
                fprintf(stderr, "Error: Could not open task2primes.txt for writing.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            printf("\nPrime numbers strictly less than %d are:\n", n);
        }

        for (int i = 2; i < n; i++) {
            if (final_map[i]) {
                if (n < 100) {
                    printf("%d ", i);
                } else {
                    fprintf(file, "%d\n", i);
                }
            }
        }

        if (file != NULL) {
            fclose(file);
        } else {
            printf("\n");
        }

        free(final_map);
        if (all_primes_gathered) free(all_primes_gathered);
        free(recvcounts);
        free(displs);
    }

    // Synchronize to record true overall program completion
    MPI_Barrier(MPI_COMM_WORLD);
    double end_total = MPI_Wtime();
    double local_total_time = end_total - start_total;

    /* =========================================================
     * GLOBAL REDUCTION FOR RIGOROUS AMDAHL METRICS
     * ========================================================= */
    double parallel_time = 0.0;
    MPI_Reduce(&local_parallel_duration, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double total_time = 0.0;
    MPI_Reduce(&local_total_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Rigorous Amdahl decomposition
        double serial_time = total_time - parallel_time;
        if (serial_time < 0.0) serial_time = 0.0;

        double p = parallel_time / total_time;
        double s = serial_time / total_time;
        int total_workers = size * num_threads;

        double theoretical_speedup = 1.0 / (s + (p / (double)total_workers));
        double max_theoretical_speedup = (s > 0.0) ? (1.0 / s) : 0.0;

        printf("\n========================================\n");
        printf("TASK 2 - HYBRID AMDAHL PERFORMANCE ANALYSIS\n");
        printf("========================================\n");
        printf("n: %d\n", n);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Total computational workers: %d\n", total_workers);
        printf("----------------------------------------\n");
        printf("Parallel phase time (Tp): %f seconds\n", parallel_time);
        printf("Serial/fabric time (Ts): %f seconds\n", serial_time);
        printf("Total execution time: %f seconds\n", total_time);
        printf("----------------------------------------\n");
        printf("Parallel fraction (p): %f\n", p);
        printf("Serial fraction (s): %f\n", s);
        printf("Check s + p: %f\n", s + p);
        printf("----------------------------------------\n");
        printf("Theoretical Amdahl speedup (%d workers): %f\n", total_workers, theoretical_speedup);
        printf("Maximum theoretical speedup (1/s): %f\n", max_theoretical_speedup);
        printf("========================================\n");
    }

    free(thread_times);
    MPI_Finalize();
    return 0;
}