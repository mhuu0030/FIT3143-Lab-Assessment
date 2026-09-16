/**
 * FIT3143 Parallel Computing - Lab 2, Task 2 (Alternative Implementation)
 * Hybrid Open MPI + OpenMP with Naive Static Block Partitioning.
 *
 * Compilation:
 *   mpicc -Wall -O2 -fopenmp task2_alt.c -o task2_alt -lm
 *
 * Execution:
 *   mpirun -np <num_procs> ./task2_alt <n> [threads_per_process]
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

bool is_prime(int k) {
    if (k <= 1) return false;
    if (k == 2) return true;
    if (k % 2 == 0) return false;

    int limit = (int)sqrt((double)k);
    for (int i = 3; i <= limit; i += 2) {
        if (k % i == 0) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    int num_threads = 4;

    double t_start_total = 0.0, t_end_total = 0.0;
    double t_start_comp = 0.0, t_end_comp = 0.0;

    if (rank == 0) {
        t_start_total = MPI_Wtime();
        if (argc < 2) {
            fprintf(stderr, "Usage: mpirun -np <p> %s <n> [threads]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
        if (argc >= 3) num_threads = atoi(argv[2]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    omp_set_num_threads(num_threads);

    // --- Naive Static Block Partitioning ---
    // Divide the numerical range [3, n) into contiguous blocks per process
    long long total_range = (long long)n - 3;
    int block_start = 3 + (int)((rank * total_range) / size);
    int block_end = 3 + (int)(((rank + 1) * total_range) / size);
    if (rank == size - 1) block_end = n;

    // Ensure block_start starts on an odd number
    if (block_start % 2 == 0) block_start++;

    int local_capacity = (block_end - block_start) / 2 + 2;
    int *local_primes = (int *)malloc(local_capacity * sizeof(int));
    bool *local_flags = (bool *)calloc(local_capacity, sizeof(bool));

    MPI_Barrier(MPI_COMM_WORLD);
    t_start_comp = MPI_Wtime();

    // Naive OpenMP static schedule over contiguous block
    #pragma omp parallel for schedule(static)
    for (int i = block_start; i < block_end; i += 2) {
        if (is_prime(i)) {
            int idx = (i - block_start) / 2;
            local_flags[idx] = true;
        }
    }

    int local_count = 0;
    for (int i = block_start; i < block_end; i += 2) {
        int idx = (i - block_start) / 2;
        if (local_flags[idx]) {
            local_primes[local_count++] = i;
        }
    }

    t_end_comp = MPI_Wtime();
    free(local_flags);

    // --- MPI Gather via Gatherv ---
    int *recvcounts = NULL, *displs = NULL, *all_primes = NULL;
    int total_gathered = 0;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        total_gathered = recvcounts[0];
        for (int p = 1; p < size; p++) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
            total_gathered += recvcounts[p];
        }
        all_primes = (int *)malloc(total_gathered * sizeof(int));
    }

    MPI_Gatherv(local_primes, local_count, MPI_INT,
                all_primes, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    free(local_primes);

    if (rank == 0) {
        // In contiguous block partitioning, blocks arrive ordered by rank,
        // so output directly without a full array reconstruct
        FILE *f = fopen("task2_alt_primes.txt", "w");
        if (f && n > 2) fprintf(f, "2\n");
        for (int i = 0; i < total_gathered; i++) {
            if (f) fprintf(f, "%d\n", all_primes[i]);
        }
        if (f) fclose(f);

        free(all_primes);
        free(recvcounts);
        free(displs);

        t_end_total = MPI_Wtime();
        printf("[task2_alt] Procs: %d, Threads: %d, N: %d, CompTime: %f s, TotalTime: %f s\n",
               size, num_threads, n, (t_end_comp - t_start_comp), (t_end_total - t_start_total));
    }

    MPI_Finalize();
    return 0;
}