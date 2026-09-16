/**
 * FIT3143 Parallel Computing - Lab 2, Task 1
 * Prime search using Open MPI.
 *
 * Workload distribution:
 * Fine-grained cyclic distribution.
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
    if (k <= 1)
        return false;

    if (k == 2)
        return true;

    if (k % 2 == 0)
        return false;

    int limit =
        (int)sqrt((double)k);


    for (int i = 3;
         i <= limit;
         i += 2)
    {
        if (k % i == 0)
            return false;
    }


    return true;
}


/**
 * Comparison function for qsort().
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
     * Root reads n
     * ---------------------------------------------------------
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
     * Synchronise before total timing.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    double start_time =
        MPI_Wtime();


    /*
     * Disseminate n.
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
            local_capacity *
            sizeof(int)
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
     * Start computation timer
     * ---------------------------------------------------------
     */

    double start_comp =
        MPI_Wtime();


    /* =========================================================
     * FINE-GRAINED CYCLIC DISTRIBUTION
     * =========================================================
     *
     * Individual odd candidates are assigned cyclically.
     *
     * Example with 3 MPI processes:
     *
     * Rank 0:
     * 3, 9, 15, 21, 27, ...
     *
     * Rank 1:
     * 5, 11, 17, 23, 29, ...
     *
     * Rank 2:
     * 7, 13, 19, 25, 31, ...
     *
     * Every rank receives values spread throughout
     * the whole search range.
     * =========================================================
     */


    /*
     * First odd candidate belonging to this rank.
     */
    long long first =
        3 +
        (2LL * rank);


    /*
     * Distance between consecutive candidates handled
     * by the same process.
     */
    long long step =
        2LL *
        num_processes;


    /*
     * Search every num_processes-th odd candidate.
     */
    for (long long candidate = first;
         candidate < n;
         candidate += step)
    {
        if (is_prime((int)candidate))
        {
            /*
             * Expand local array if necessary.
             */
            if (local_count ==
                local_capacity)
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


                local_primes =
                    temp;
            }


            local_primes[local_count] =
                (int)candidate;

            local_count++;
        }
    }


    /*
     * End computational timing.
     */
    double end_comp =
        MPI_Wtime();


    double local_comp_time =
        end_comp -
        start_comp;


    /* ---------------------------------------------------------
     * Gather result counts
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
     * Root prepares receive buffer
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
         * Extra location for prime number 2.
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
     * Gather local prime arrays
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
     * Root sorts and writes final results
     * ---------------------------------------------------------
     */

    if (rank == 0)
    {
        /*
         * Add 2 separately.
         */
        if (n > 2)
        {
            all_primes[total_count] =
                2;

            total_count++;
        }


        /*
         * Cyclic distribution means the gathered
         * arrays are not globally sorted.
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
     * Finish total execution timing
     * ---------------------------------------------------------
     */

    MPI_Barrier(
        MPI_COMM_WORLD
    );


    double local_total_time =
        MPI_Wtime() -
        start_time;


    double total_time = 0.0;


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
     * Gather individual computation times
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
     * Root prints timing results
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
            "Workload distribution: Fine-grained cyclic\n"
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
     * Free memory
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


    MPI_Finalize();

    return 0;
}