/**
 * FIT3143 Parallel Computing - Lab 2, Task 1
 * Prime search using Open MPI.
 *
 * Workload distribution:
 * Contiguous block distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>


/**
 * Checks whether k is a prime number.
 *
 * @param k Integer to test.
 * @return true if prime, false otherwise.
 */
bool is_prime(int k)
{
    // 0 and 1 are not prime
    if (k <= 1)
        return false;

    // 2 is the only even prime
    if (k == 2)
        return true;

    // Eliminate all other even numbers
    if (k % 2 == 0)
        return false;

    // Only need to test factors up to sqrt(k)
    int limit = (int)sqrt((double)k);

    // Test odd divisors only
    for (int i = 3; i <= limit; i += 2)
    {
        if (k % i == 0)
            return false;
    }

    return true;
}


/**
 * Comparison function used by qsort().
 */
int compare_ints(const void *a, const void *b)
{
    int x = *(const int *)a;
    int y = *(const int *)b;

    if (x < y)
        return -1;

    if (x > y)
        return 1;

    return 0;
}


int main(int argc, char *argv[])
{
    int rank;
    int num_processes;

    int n = 0;
    int valid_input = 1;


    /* ---------------------------------------------------------
     * Initialise MPI
     * ---------------------------------------------------------
     */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &num_processes
    );


    /* ---------------------------------------------------------
     * Root process reads n from command line
     * ---------------------------------------------------------
     *
     * Example:
     *
     * mpirun -np 8 ./task1_contiguous 50000000
     */

    if (rank == 0)
    {
        if (argc != 2)
        {
            printf(
                "Usage: %s <n>\n",
                argv[0]
            );

            printf(
                "Example: %s 50000000\n",
                argv[0]
            );

            valid_input = 0;
        }
        else
        {
            n = atoi(argv[1]);

            if (n <= 0)
            {
                printf(
                    "Error: Please enter a valid positive integer.\n"
                );

                valid_input = 0;
            }
        }
    }


    /*
     * Tell every process whether the input is valid.
     */
    MPI_Bcast(
        &valid_input,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    if (!valid_input)
    {
        MPI_Finalize();
        return 1;
    }


    /*
     * Synchronise processes before starting
     * overall wall-clock timing.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();


    /*
     * Root distributes n to all MPI processes.
     */
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    /* ---------------------------------------------------------
     * Local result storage
     * ---------------------------------------------------------
     */

    int local_capacity = 1024;
    int local_count = 0;

    int *local_primes =
        (int *)malloc(
            local_capacity * sizeof(int)
        );


    if (local_primes == NULL)
    {
        printf(
            "Process %d: Memory allocation failed.\n",
            rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }


    /* ---------------------------------------------------------
     * Begin computation timing
     * ---------------------------------------------------------
     */

    double start_comp = MPI_Wtime();


    /* =========================================================
     * CONTIGUOUS BLOCK DISTRIBUTION
     * =========================================================
     *
     * All odd candidates are divided into continuous sections.
     *
     * Example:
     *
     * Odd candidates:
     *
     * 3 5 7 9 11 13 15 17 19 21 23 25 ...
     *
     * With 3 MPI processes:
     *
     * Rank 0:
     * 3 5 7 9 ...
     *
     * Rank 1:
     * middle section
     *
     * Rank 2:
     * upper section
     *
     * Each process receives approximately the same NUMBER
     * of candidates, but not necessarily the same amount of
     * computational work.
     * =========================================================
     */


    /*
     * Number of odd candidate numbers from 3 to n - 1.
     *
     * Example:
     *
     * n = 10
     *
     * candidates:
     * 3, 5, 7, 9
     *
     * total = 4
     */

    long long total_odd_candidates = 0;

    if (n > 3)
    {
        total_odd_candidates =
            (n - 2) / 2;
    }


    /*
     * Basic number of candidates per MPI process.
     */
    long long base =
        total_odd_candidates /
        num_processes;


    /*
     * Candidates remaining after equal division.
     */
    long long extra =
        total_odd_candidates %
        num_processes;


    /*
     * First 'extra' ranks receive one additional candidate.
     */
    long long my_count =
        base +
        (rank < extra ? 1 : 0);


    /*
     * Determine the first candidate index belonging
     * to this process.
     */
    long long start_index =
        rank * base +
        (rank < extra ? rank : extra);


    /*
     * Convert the candidate index into the
     * actual odd number.
     *
     * index 0 -> 3
     * index 1 -> 5
     * index 2 -> 7
     */
    long long first =
        3 +
        (2LL * start_index);


    /*
     * Search this process's continuous section.
     */
    for (long long j = 0;
         j < my_count;
         j++)
    {
        long long candidate =
            first +
            (2LL * j);


        if (is_prime((int)candidate))
        {
            /*
             * Expand local result array if necessary.
             */
            if (local_count == local_capacity)
            {
                local_capacity *= 2;

                int *temp =
                    (int *)realloc(
                        local_primes,
                        local_capacity *
                        sizeof(int)
                    );


                if (temp == NULL)
                {
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


            local_primes[local_count] =
                (int)candidate;

            local_count++;
        }
    }


    /*
     * End computational timing.
     */
    double end_comp = MPI_Wtime();

    double local_comp_time =
        end_comp - start_comp;


    /* ---------------------------------------------------------
     * Gather number of primes found by each process
     * ---------------------------------------------------------
     */

    int *recv_counts = NULL;


    if (rank == 0)
    {
        recv_counts =
            (int *)malloc(
                num_processes *
                sizeof(int)
            );


        if (recv_counts == NULL)
        {
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


    /* ---------------------------------------------------------
     * Root calculates displacements
     * ---------------------------------------------------------
     */

    int *displacements = NULL;
    int *all_primes = NULL;

    int total_count = 0;


    if (rank == 0)
    {
        displacements =
            (int *)malloc(
                num_processes *
                sizeof(int)
            );


        if (displacements == NULL)
        {
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
             i++)
        {
            total_count +=
                recv_counts[i];


            if (i > 0)
            {
                displacements[i] =
                    displacements[i - 1] +
                    recv_counts[i - 1];
            }
        }


        /*
         * +1 leaves space for prime number 2.
         */
        all_primes =
            (int *)malloc(
                (total_count + 1) *
                sizeof(int)
            );


        if (all_primes == NULL)
        {
            printf(
                "Error: Memory allocation failed.\n"
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }
    }


    /* ---------------------------------------------------------
     * Gather variable-sized local prime arrays
     * ---------------------------------------------------------
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


    /* ---------------------------------------------------------
     * Root combines, sorts and outputs
     * ---------------------------------------------------------
     */

    if (rank == 0)
    {
        /*
         * 2 is the only even prime.
         */
        if (n > 2)
        {
            all_primes[total_count] = 2;
            total_count++;
        }


        /*
         * Sort final result.
         */
        qsort(
            all_primes,
            total_count,
            sizeof(int),
            compare_ints
        );


        FILE *file = NULL;


        if (n >= 100)
        {
            file =
                fopen(
                    "task1primes.txt",
                    "w"
                );


            if (file == NULL)
            {
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
        }
        else
        {
            printf(
                "Prime numbers strictly less than %d are:\n",
                n
            );
        }


        for (int i = 0;
             i < total_count;
             i++)
        {
            if (n < 100)
            {
                printf(
                    "%d ",
                    all_primes[i]
                );
            }
            else
            {
                fprintf(
                    file,
                    "%d\n",
                    all_primes[i]
                );
            }
        }


        if (file != NULL)
        {
            fclose(file);
        }
        else
        {
            printf("\n");
        }
    }


    /* ---------------------------------------------------------
     * Finish overall wall-clock timing
     * ---------------------------------------------------------
     *
     * Other ranks wait while rank 0 completes sorting
     * and file writing.
     */

    MPI_Barrier(MPI_COMM_WORLD);


    double local_total_time =
        MPI_Wtime() -
        start_time;


    double total_time = 0.0;


    /*
     * Overall parallel runtime is determined by
     * the slowest MPI process.
     */
    MPI_Reduce(
        &local_total_time,
        &total_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );


    /* ---------------------------------------------------------
     * Gather individual process computation times
     * ---------------------------------------------------------
     */

    double *process_times = NULL;


    if (rank == 0)
    {
        process_times =
            (double *)malloc(
                num_processes *
                sizeof(double)
            );


        if (process_times == NULL)
        {
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
        &local_comp_time,
        1,
        MPI_DOUBLE,

        process_times,
        1,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );


    /* ---------------------------------------------------------
     * Print timing results
     * ---------------------------------------------------------
     */

    if (rank == 0)
    {
        double max_comp_time = 0.0;


        printf(
            "\nNumber of MPI processes: %d\n",
            num_processes
        );


        printf(
            "Workload distribution: Contiguous block\n"
        );


        for (int i = 0;
             i < num_processes;
             i++)
        {
            printf(
                "Process %d computational time: %f seconds\n",
                i,
                process_times[i]
            );


            if (process_times[i] >
                max_comp_time)
            {
                max_comp_time =
                    process_times[i];
            }
        }


        printf(
            "Computational time (slowest process): %f seconds\n",
            max_comp_time
        );


        printf(
            "Total execution time: %f seconds\n",
            total_time
        );
    }


    /* ---------------------------------------------------------
     * Free allocated memory
     * ---------------------------------------------------------
     */

    free(local_primes);


    if (rank == 0)
    {
        free(recv_counts);
        free(displacements);
        free(all_primes);
        free(process_times);
    }


    /* ---------------------------------------------------------
     * Finish MPI
     * ---------------------------------------------------------
     */

    MPI_Finalize();

    return 0;
}