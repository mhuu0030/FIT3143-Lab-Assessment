/**
 * FIT3143 Parallel Computing - Lab 2, Task 2
 * Hybrid Open MPI + OpenMP Prime Search (strictly less than n)
 *
 * Compilation:
 *   mpicc -Wall -O2 -fopenmp task2.c -o task2 -lm
 *
 * Execution:
 *   mpirun -np <num_procs> ./task2 <n> [threads_per_process]
 *   Example: mpirun -np 4 ./task2 10000000 4
 */

#define _POSIX_C_SOURCE 199309L // Expose CLOCK_MONOTONIC and CLOCK_THREAD_CPUTIME_ID

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization and eliminates even divisors.
 *
 * @param k The integer to check for primality.
 * @return true if k is prime, false otherwise.
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

    // Initialize MPI with thread support (MPI calls made by the main thread only)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    int num_threads = 4; // Default OpenMP thread count per MPI process

    // High-resolution timers for Amdahl's Law serial vs parallel decomposition
    double t_start_total = 0.0, t_end_total = 0.0;
    double t_start_comp = 0.0, t_end_comp = 0.0;
    double t_start_comm = 0.0, t_end_comm = 0.0;

    // Root process setup and command-line argument validation
    if (rank == 0) {
        t_start_total = MPI_Wtime();

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

    // Disseminate n and num_threads to all MPI worker ranks
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Configure thread pool for OpenMP
    omp_set_num_threads(num_threads);

    // Distributed Workload: Cyclic stride across odd integers
    int mpi_step = 2 * size;
    int mpi_first = 3 + (2 * rank);

    // Local buffer capacity for primes handled by this process
    int local_capacity = (n > mpi_first) ? ((n - mpi_first) / mpi_step + 2) : 1;
    bool *local_flags = (bool *)calloc(local_capacity, sizeof(bool));
    int *local_primes = (int *)malloc(local_capacity * sizeof(int));

    if (local_flags == NULL || local_primes == NULL) {
        fprintf(stderr, "[Rank %d] Error: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Thread-level profiling array (from task3.c)
    double *thread_times = (double *)calloc(num_threads, sizeof(double));

    // Barrier to synchronize start of computation across nodes
    MPI_Barrier(MPI_COMM_WORLD);
    t_start_comp = MPI_Wtime();

    // ==========================================
    // HYBRID OPENMP PARALLEL REGION
    // ==========================================
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        struct timespec t_start_thread, t_end_thread;

        // Measure individual thread CPU time
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_start_thread);

        // Dynamic work-sharing schedule to absorb non-linear O(sqrt(k)) variance
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

    // Sequentially pack discovered primes into local_primes buffer
    int local_count = 0;
    for (int i = mpi_first; i < n; i += mpi_step) {
        int idx = (i - mpi_first) / mpi_step;
        if (local_flags[idx]) {
            local_primes[local_count++] = i;
        }
    }

    t_end_comp = MPI_Wtime();
    free(local_flags);

    // ==========================================
    // COMMUNICATION & GATHER PHASE (MPI)
    // ==========================================
    t_start_comm = MPI_Wtime();

    int *recvcounts = NULL;
    int *displs = NULL;
    int *all_primes_gathered = NULL;
    int total_gathered_primes = 0;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    // Gather count of primes found by each MPI rank
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
            fprintf(stderr, "[Rank 0] Error: Memory allocation failed for gather buffer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Variable gather of local prime arrays into root rank
    MPI_Gatherv(local_primes, local_count, MPI_INT,
                all_primes_gathered, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    t_end_comm = MPI_Wtime();
    free(local_primes);

    // Sequential output of thread timings per rank
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            printf("[Rank %d] OpenMP Thread CPU Times:\n", rank);
            for (int t = 0; t < num_threads; t++) {
                printf("   -- Thread %d CPU time: %f s\n", t, thread_times[t]);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(thread_times);

    // ==========================================
    // SERIAL I/O & ORDERING VERIFICATION (Rank 0)
    // ==========================================
    if (rank == 0) {
        bool *final_map = (bool *)calloc(n, sizeof(bool));
        if (final_map == NULL) {
            fprintf(stderr, "[Rank 0] Error: Memory allocation failed for output sorting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Account for prime 2
        if (n > 2) final_map[2] = true;

        // Map gathered primes into sequential index space
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
            printf("\nWriting sorted primes to task2primes.txt...\n");
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

        t_end_total = MPI_Wtime();

        // Print Amdahl's Law metrics for Task 3 benchmarking
        double comp_time = t_end_comp - t_start_comp;
        double comm_time = t_end_comm - t_start_comm;
        double total_time = t_end_total - t_start_total;
        double serial_time = total_time - comp_time;

        printf("\n================ Performance Metrics ================\n");
        printf("MPI Processes (Nodes)        : %d\n", size);
        printf("OpenMP Threads per Process   : %d\n", num_threads);
        printf("Total Computational Workers  : %d\n", size * num_threads);
        printf("Search Space Bound (n)       : %d\n", n);
        printf("Parallel Computation Time    : %f seconds\n", comp_time);
        printf("MPI Communication Time       : %f seconds\n", comm_time);
        printf("Serial Setup, I/O & Overhead : %f seconds\n", serial_time);
        printf("Total Wall-Clock Time        : %f seconds\n", total_time);
        printf("=====================================================\n");
    }

    MPI_Finalize();
    return 0;
}