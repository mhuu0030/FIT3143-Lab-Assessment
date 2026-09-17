/**
 * FIT3143 Parallel Computing - Lab 2, Task 3
 *
 * Task 1 Open MPI prime search instrumented for
 * Amdahl's Law performance analysis.
 *
 * Workload distribution:
 * Block round-robin / block-cyclic.
 *
 * Task 3 measurements:
 *
 * T_total    = overall wall-clock execution time
 * T_parallel = wall-clock duration of parallel prime-search phase
 * T_serial   = T_total - T_parallel
 *
 * p = T_parallel / T_total
 * s = T_serial   / T_total
 *
 * Amdahl:
 *
 * Speedup(N) = 1 / (s + p/N)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

#define BLOCK_SIZE 1000


/**
 * Check whether k is prime.
 */
bool is_prime(int k)
{
    if (k <= 1)
        return false;

    if (k == 2)
        return true;

    if (k % 2 == 0)
        return false;

    int limit = (int)sqrt((double)k);

    for (int i = 3; i <= limit; i += 2)
    {
        if (k % i == 0)
        {
            return false;
        }
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


    /* =========================================================
     * INITIALISE MPI
     * =========================================================
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


    /* =========================================================
     * ROOT READS N
     * =========================================================
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
     * Tell every process whether input is valid.
     *
     * This happens before our total timer,
     * same as the original Task 1 design.
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


    /* =========================================================
     * TOTAL EXECUTION TIMER
     * =========================================================
     *
     * Synchronise first so all processes begin the measured
     * program at approximately the same point.
     */

    MPI_Barrier(MPI_COMM_WORLD);

    double start_total = MPI_Wtime();


    /*
     * n communication is therefore INCLUDED in total time.
     */
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    /* =========================================================
     * LOCAL STORAGE
     * =========================================================
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


    /* =========================================================
     * TASK 3: PARALLEL PHASE MEASUREMENT
     * =========================================================
     *
     * The barrier creates a common starting point.
     *
     * This is NOT required for the normal Task 1 algorithm.
     * It is added here specifically so we can measure the
     * wall-clock duration of the complete parallel phase.
     */

    MPI_Barrier(MPI_COMM_WORLD);

    double start_parallel_phase =
        MPI_Wtime();


    /*
     * Also measure how long this individual rank spends
     * performing actual computation.
     *
     * This is useful for workload balance diagnostics.
     */
    double start_local_compute =
        MPI_Wtime();


    /* =========================================================
     * BLOCK-CYCLIC WORKLOAD DISTRIBUTION
     * =========================================================
     */

    for (long long block = rank;
         ;
         block += num_processes)
    {
        /*
         * Beginning and end of this block.
         */
        long long start =
            (block * BLOCK_SIZE) + 1;

        long long end =
            start + BLOCK_SIZE - 1;


        /*
         * Stop when this rank has no more blocks.
         */
        if (start >= n)
        {
            break;
        }


        /*
         * Find primes strictly less than n.
         */
        if (end >= n)
        {
            end = n - 1;
        }


        /*
         * 2 is handled separately.
         */
        if (start < 3)
        {
            start = 3;
        }


        /*
         * Skip even candidate numbers.
         */
        if (start % 2 == 0)
        {
            start++;
        }


        /*
         * Check only odd candidates.
         */
        for (long long i = start;
             i <= end;
             i += 2)
        {
            if (is_prime((int)i))
            {
                /*
                 * Expand result array if necessary.
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
                    (int)i;

                local_count++;
            }
        }
    }


    /*
     * Individual rank computation ends here.
     */
    double end_local_compute =
        MPI_Wtime();


    double local_compute_time =
        end_local_compute -
        start_local_compute;


    /*
     * =========================================================
     * IMPORTANT TASK 3 BARRIER
     * =========================================================
     *
     * Faster processes wait here until the slowest process
     * completes its prime-search workload.
     *
     * Therefore the parallel phase represents how long the
     * whole application must wait for parallel computation
     * to complete.
     */

    MPI_Barrier(MPI_COMM_WORLD);


    double end_parallel_phase =
        MPI_Wtime();


    double local_parallel_phase_time =
        end_parallel_phase -
        start_parallel_phase;


    /* =========================================================
     * COLLECT RESULTS
     * =========================================================
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


    /*
     * Fabric / MPI communication.
     *
     * This is NOT part of T_parallel.
     *
     * Because it is inside T_total, it will automatically
     * be included in T_serial when:
     *
     * T_serial = T_total - T_parallel
     */
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
         * +1 for prime number 2.
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


    /*
     * Fabric / communication time is again outside
     * the parallel prime-search phase.
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


    /* =========================================================
     * ROOT SERIAL WORK
     * =========================================================
     */

    if (rank == 0)
    {
        /*
         * Add prime number 2.
         */
        if (n > 2)
        {
            all_primes[total_count] =
                2;

            total_count++;
        }


        /*
         * Sorting is root-only work.
         *
         * Therefore it becomes part of the serial fraction.
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


        /*
         * File output is also root-only / serial work.
         */
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


    /* =========================================================
     * END TOTAL TIMER
     * =========================================================
     *
     * Rank 0 performs the final sorting and file output,
     * so its elapsed time represents the complete program
     * path that we want to analyse.
     */

    double end_total =
        MPI_Wtime();


    double local_total_time =
        end_total -
        start_total;


    /* =========================================================
     * TASK 3 DIAGNOSTIC COMMUNICATION
     *
     * Everything below happens AFTER the measured program.
     * It therefore does not affect T_total.
     * =========================================================
     */


    /*
     * Obtain the largest parallel-phase duration.
     *
     * We use MPI_MAX because the parallel program cannot
     * progress faster than its slowest participating rank.
     */
    double parallel_time = 0.0;


    MPI_Reduce(
        &local_parallel_phase_time,
        &parallel_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );


    /*
     * Gather individual computation times for diagnostics.
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
        &local_compute_time,
        1,
        MPI_DOUBLE,

        process_times,
        1,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );


    /*
     * Gather total elapsed time from all processes.
     *
     * Rank 0 performs sorting/output and should normally
     * have the longest total duration.
     *
     * MPI_MAX makes the definition robust.
     */
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


    /* =========================================================
     * TASK 3 AMDahl ANALYSIS
     * =========================================================
     */

    if (rank == 0)
    {
        /*
         * Anything outside the measured parallel prime-search
         * phase is classified as the serial fraction.
         *
         * This therefore includes:
         *
         * - MPI communication / fabric time
         * - result gathering
         * - root calculations
         * - sorting
         * - file output
         */
        double serial_time =
            total_time -
            parallel_time;


        /*
         * Prevent tiny floating-point errors producing
         * a negative result.
         */
        if (serial_time < 0.0)
        {
            serial_time = 0.0;
        }


        double parallel_fraction =
            parallel_time /
            total_time;


        double serial_fraction =
            serial_time /
            total_time;


        /*
         * Amdahl's Law:
         *
         * S(N) = 1 / (s + p/N)
         */
        double theoretical_speedup =
            1.0 /
            (
                serial_fraction +
                (
                    parallel_fraction /
                    num_processes
                )
            );


        /*
         * Maximum possible Amdahl speedup as N approaches
         * infinity:
         *
         * S_max = 1 / s
         */
        double max_theoretical_speedup =
            0.0;


        if (serial_fraction > 0.0)
        {
            max_theoretical_speedup =
                1.0 /
                serial_fraction;
        }


        printf(
            "\n========================================\n"
        );

        printf(
            "TASK 3 - AMDahl PERFORMANCE ANALYSIS\n"
        );

        printf(
            "========================================\n"
        );


        printf(
            "n: %d\n",
            n
        );


        printf(
            "MPI processes: %d\n",
            num_processes
        );


        printf(
            "Block size: %d\n",
            BLOCK_SIZE
        );


        printf(
            "\nIndividual rank computation times:\n"
        );


        double max_local_compute = 0.0;


        for (int i = 0;
             i < num_processes;
             i++)
        {
            printf(
                "Process %d computation time: %f seconds\n",
                i,
                process_times[i]
            );


            if (process_times[i] >
                max_local_compute)
            {
                max_local_compute =
                    process_times[i];
            }
        }


        printf(
            "\n----------------------------------------\n"
        );

        printf(
            "Slowest rank computation time: %f seconds\n",
            max_local_compute
        );


        printf(
            "Parallel phase time (Tp): %f seconds\n",
            parallel_time
        );


        printf(
            "Serial/fabric time (Ts): %f seconds\n",
            serial_time
        );


        printf(
            "Total execution time: %f seconds\n",
            total_time
        );


        printf(
            "\n----------------------------------------\n"
        );


        printf(
            "Parallel fraction (p): %f\n",
            parallel_fraction
        );


        printf(
            "Serial fraction (s): %f\n",
            serial_fraction
        );


        printf(
            "Check s + p: %f\n",
            serial_fraction +
            parallel_fraction
        );


        printf(
            "\n----------------------------------------\n"
        );


        printf(
            "Theoretical Amdahl speedup for %d processes: %f\n",
            num_processes,
            theoretical_speedup
        );


        printf(
            "Maximum theoretical speedup (1/s): %f\n",
            max_theoretical_speedup
        );


        printf(
            "========================================\n"
        );
    }


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