/**
 * FIT3143 Parallel Computing - Lab 2, Task 1
 * Parallel code to find prime numbers strictly less than an integer n
 * using Open MPI.
 *
 * Workload distribution:
 * Block round-robin / block-cyclic distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

#define BLOCK_SIZE 1000

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization to eliminate unnecessary computations.
 *
 * @param k The integer to check for primality.
 * @return true if k is prime, false otherwise.
 */
bool is_prime(int k) {

    // 0 and 1 are not prime numbers
    if (k <= 1) return false;

    // 2 is the only even prime number
    if (k == 2) return true;

    // Eliminate all other even numbers immediately
    if (k % 2 == 0) return false;

    // Only need to check factors up to sqrt(k)
    int limit = (int)sqrt((double)k);

    // Check odd divisors only
    for (int i = 3; i <= limit; i += 2) {
        if (k % i == 0) {
            return false;
        }
    }

    return true;
}


/**
 * Comparison function used by qsort().
 */
int compare_ints(const void *a, const void *b) {

    int x = *(const int *)a;
    int y = *(const int *)b;

    if (x < y) return -1;
    if (x > y) return 1;

    return 0;
}


int main(int argc, char *argv[]) {

    int rank;
    int num_processes;
    int n = 0;
    int valid_input = 1;

    /*
     * Start MPI environment.
     */
    MPI_Init(&argc, &argv);

    /*
     * Get:
     * rank          = ID of this MPI process
     * num_processes = total number of MPI processes
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);


    /*
     * Only the root process reads n.
     *
     * Example:
     * srun ./task1 10000000
     */
    if (rank == 0) {

        if (argc != 2) {

            printf("Usage: %s <n>\n", argv[0]);
            printf("Example: %s 10000000\n", argv[0]);

            valid_input = 0;

        } else {

            n = atoi(argv[1]);

            if (n <= 0) {

                printf("Error: Please enter a valid positive integer.\n");
                valid_input = 0;
            }
        }
    }


    /*
     * Tell every MPI process whether the input was valid.
     */
    MPI_Bcast(
        &valid_input,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    if (!valid_input) {

        MPI_Finalize();
        return 1;
    }


    /*
     * Synchronize all processes before timing.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    /*
     * Start total execution time.
     *
     * This is before MPI_Bcast(n), so communication is included
     * in the total MPI execution time.
     */
    double start_time = MPI_Wtime();


    /*
     * Root process sends n to every MPI process.
     */
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    /*
     * Local array used by each process to store the primes
     * that it discovers.
     *
     * Start with a small array and increase it when required.
     */
    int local_capacity = 1024;
    int local_count = 0;

    int *local_primes =
        (int *)malloc(local_capacity * sizeof(int));


    if (local_primes == NULL) {

        printf(
            "Process %d: Memory allocation failed.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    /*
     * Start measuring the computational section.
     */
    double start_comp = MPI_Wtime();


    /*
     * ==========================================================
     * BLOCK ROUND-ROBIN / BLOCK-CYCLIC WORKLOAD DISTRIBUTION
     * ==========================================================
     *
     * Example:
     *
     * BLOCK_SIZE = 10
     * num_processes = 3
     *
     * Rank 0:
     *      1 - 10
     *     31 - 40
     *     61 - 70
     *     ...
     *
     * Rank 1:
     *     11 - 20
     *     41 - 50
     *     71 - 80
     *     ...
     *
     * Rank 2:
     *     21 - 30
     *     51 - 60
     *     81 - 90
     *     ...
     *
     *
     * Each MPI process starts from its own block number:
     *
     * block = rank
     *
     * and jumps by:
     *
     * block += num_processes
     *
     * This distributes blocks in round-robin order.
     */

    for (long long block = rank;
         ;
         block += num_processes) {

        /*
         * Calculate the beginning and end of this block.
         */
        long long start =
            (block * BLOCK_SIZE) + 1;

        long long end =
            start + BLOCK_SIZE - 1;


        /*
         * Stop when this process has gone beyond n.
         */
        if (start >= n) {
            break;
        }


        /*
         * The program finds primes strictly LESS THAN n.
         */
        if (end >= n) {
            end = n - 1;
        }


        /*
         * 2 will be handled separately by the root later.
         *
         * Start checking from at least 3.
         */
        if (start < 3) {
            start = 3;
        }


        /*
         * We already know all even numbers > 2 are not prime.
         *
         * Therefore, make sure the first candidate is odd.
         */
        if (start % 2 == 0) {
            start++;
        }


        /*
         * Search only odd candidates in this block.
         */
        for (long long i = start;
             i <= end;
             i += 2) {

            if (is_prime((int)i)) {

                /*
                 * If local array becomes full,
                 * double its size.
                 */
                if (local_count == local_capacity) {

                    local_capacity *= 2;

                    int *temp =
                        (int *)realloc(
                            local_primes,
                            local_capacity * sizeof(int)
                        );


                    if (temp == NULL) {

                        printf(
                            "Process %d: Memory reallocation failed.\n",
                            rank
                        );

                        free(local_primes);

                        MPI_Abort(
                            MPI_COMM_WORLD,
                            1
                        );
                    }


                    local_primes = temp;
                }


                /*
                 * Save the prime found by this process.
                 */
                local_primes[local_count] = (int)i;

                local_count++;
            }
        }
    }


    /*
     * End computational timing for this MPI process.
     */
    double end_comp = MPI_Wtime();

    double local_comp_time =
        end_comp - start_comp;


    /*
     * ==========================================================
     * COLLECT RESULTS FROM ALL MPI PROCESSES
     * ==========================================================
     *
     * Different MPI processes may find different numbers
     * of prime numbers.
     *
     * First gather how many primes each process found.
     */

    int *recv_counts = NULL;

    if (rank == 0) {

        recv_counts =
            (int *)malloc(
                num_processes * sizeof(int)
            );


        if (recv_counts == NULL) {

            printf(
                "Error: Memory allocation failed.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }
    }


    MPI_Gather(
        &local_count,
        1,
        MPI_INT,

        recv_counts,
        1,
        MPI_INT,

        0,
        MPI_COMM_WORLD
    );


    /*
     * Root determines:
     *
     * 1. Total number of primes found
     * 2. Where each process's results should be placed
     */
    int *displacements = NULL;

    int total_count = 0;

    int *all_primes = NULL;


    if (rank == 0) {

        displacements =
            (int *)malloc(
                num_processes * sizeof(int)
            );


        if (displacements == NULL) {

            printf(
                "Error: Memory allocation failed.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }


        displacements[0] = 0;


        for (int i = 0;
             i < num_processes;
             i++) {

            total_count += recv_counts[i];

            if (i > 0) {

                displacements[i] =
                    displacements[i - 1]
                    + recv_counts[i - 1];
            }
        }


        /*
         * +1 leaves space for prime number 2.
         */
        all_primes =
            (int *)malloc(
                (total_count + 1)
                * sizeof(int)
            );


        if (all_primes == NULL) {

            printf(
                "Error: Memory allocation failed.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }
    }


    /*
     * Gather variable-sized prime arrays from all MPI processes.
     */
    MPI_Gatherv(
        local_primes,
        local_count,
        MPI_INT,

        all_primes,
        recv_counts,
        displacements,
        MPI_INT,

        0,
        MPI_COMM_WORLD
    );


    /*
     * ==========================================================
     * ROOT PROCESS COMBINES, SORTS AND OUTPUTS RESULTS
     * ==========================================================
     */
    if (rank == 0) {

        /*
         * 2 is the only even prime number.
         */
        if (n > 2) {

            all_primes[total_count] = 2;
            total_count++;
        }


        /*
         * Because block round-robin distribution means the
         * gathered results are not globally ordered,
         * sort all primes before output.
         */
        qsort(
            all_primes,
            total_count,
            sizeof(int),
            compare_ints
        );


        FILE *file = NULL;


        /*
         * Same output behaviour as your Week 4 programs.
         */
        if (n >= 100) {

            file =
                fopen(
                    "task1primes.txt",
                    "w"
                );


            if (file == NULL) {

                printf(
                    "Error: Could not open task1primes.txt for writing.\n"
                );

                MPI_Abort(
                    MPI_COMM_WORLD,
                    1
                );
            }


            printf(
                "Calculating complete. "
                "Output written to task1primes.txt\n"
            );

        } else {

            printf(
                "Prime numbers strictly less than %d are:\n",
                n
            );
        }


        /*
         * Output sorted prime numbers.
         */
        for (int i = 0;
             i < total_count;
             i++) {

            if (n < 100) {

                printf(
                    "%d ",
                    all_primes[i]
                );

            } else {

                fprintf(
                    file,
                    "%d\n",
                    all_primes[i]
                );
            }
        }


        if (file != NULL) {

            fclose(file);

        } else {

            printf("\n");
        }
    }


    /*
     * The overall timer ends AFTER:
     *
     * - broadcasting n
     * - computation
     * - MPI communication
     * - gathering
     * - sorting
     * - file writing
     *
     * This is important for the assignment's empirical speedup.
     */
    double end_time = MPI_Wtime();


    /*
     * Gather computational time from every MPI process.
     *
     * This occurs AFTER the overall timing so this diagnostic
     * operation does not affect the measured total execution time.
     */
    double *process_times = NULL;

    if (rank == 0) {

        process_times =
            (double *)malloc(
                num_processes
                * sizeof(double)
            );
    }


    MPI_Gather(
        &local_comp_time,
        1,
        MPI_DOUBLE,

        process_times,
        1,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );


    /*
     * Root displays timing information.
     */
    if (rank == 0) {

        double max_comp_time = 0.0;


        printf(
            "\nNumber of MPI processes: %d\n",
            num_processes
        );

        printf(
            "Block size: %d\n",
            BLOCK_SIZE
        );


        for (int i = 0;
             i < num_processes;
             i++) {

            printf(
                "Process %d computational time: %f seconds\n",
                i,
                process_times[i]
            );


            if (process_times[i] > max_comp_time) {

                max_comp_time =
                    process_times[i];
            }
        }


        /*
         * The slowest process determines how long
         * the parallel computation takes.
         */
        printf(
            "Computational time (slowest process): %f seconds\n",
            max_comp_time
        );


        printf(
            "Total execution time: %f seconds\n",
            end_time - start_time
        );
    }


    /*
     * Free local memory.
     */
    free(local_primes);


    /*
     * Root-only memory.
     */
    if (rank == 0) {

        free(recv_counts);
        free(displacements);
        free(all_primes);
        free(process_times);
    }


    /*
     * Shut down MPI.
     */
    MPI_Finalize();

    return 0;
}