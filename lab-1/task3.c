/**
 * FIT3143 Parallel Computing - Lab 1, Task 3
 * OpenMP parallel code to find prime numbers strictly less than an integer n.
 * Includes per-thread CPU computation time tracking and odd-number loop optimization.
 */

#define _POSIX_C_SOURCE 199309L // Required to expose CLOCK_MONOTONIC and CLOCK_THREAD_CPUTIME_ID

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Include OpenMP library

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization to eliminate unnecessary computations.
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
    int n;
    int num_threads;

    printf("Enter an integer n (program will find primes strictly less than n): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Error: Please enter a valid positive integer.\n");
        return 1;
    }

    // Prompt user for number of threads
    printf("Enter the number of threads: ");
    if (scanf("%d", &num_threads) != 1 || num_threads <= 0) {
        printf("Error: Please enter a valid positive number of threads.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);
    
    // This ensures file I/O setup and memory allocation are included in total time
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    FILE *file = NULL;
    if (n >= 100) {
        file = fopen("task3primes.txt", "w");
        if (file == NULL) {
            printf("Error: Could not open task3primes.txt for writing.\n");
            return 1;
        }
        printf("Calculating with OpenMP... Output will be written to task3primes.txt\n");
    } else {
        printf("Prime numbers strictly less than %d are:\n", n);
    }

    // Allocate the boolean array for primes
    bool *prime_flags = (bool *)calloc(n, sizeof(bool));
    if (prime_flags == NULL) {
        printf("Error: Memory allocation failed.\n");
        if (file) fclose(file);
        return 1;
    }
    
    // --> FIX 2: PRE-ALLOCATE THREAD TIMING ARRAY <--
    // Get the maximum number of threads OpenMP will use to size our array
    int max_threads = omp_get_max_threads();
    double *thread_times = (double *)calloc(max_threads, sizeof(double));

    if (n > 2) {
        prime_flags[2] = true;
    }

    // OPENMP PARALLEL REGION
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        struct timespec threadStart, threadEnd;

        // Start measuring CPU time for this thread
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadStart);

        // Work-sharing construct: dynamically distribute the loop iterations
        #pragma omp for schedule(dynamic, 128)
        for (int i = 3; i < n; i += 2) {
            if (is_prime(i)) {
                prime_flags[i] = true;
            }
        }

        // End measuring CPU time for this thread
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadEnd);

        // Save the time to the array INSTEAD of printing it. 
        // No #pragma omp critical needed! Lock-free and blazing fast.
        thread_times[tid] = (threadEnd.tv_sec - threadStart.tv_sec) + 
                            (threadEnd.tv_nsec - threadStart.tv_nsec) / 1e9;

    } // End of parallel region

    // --> FIX 3: PRINT THREAD TIMES SEQUENTIALLY <--
    // The parallel region is closed, so printing here doesn't bottleneck computation
    for (int i = 0; i < max_threads; i++) {
        if (thread_times[i] > 0.0) { // Only print if the thread was actually utilized
            printf("Thread %d CPU time (s): %f\n", i, thread_times[i]);
        }
    }
    
    // Clean up the profiling array
    free(thread_times);

    // Serial I/O Loop to guarantee sorted ascending order
    for (int i = 2; i < n; i++) {
        if (prime_flags[i]) {
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

    free(prime_flags); 

    // End overall wall-clock time AFTER output
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate total overall wall-clock execution time
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("\nOverall Execution time: %f seconds\n", time_taken);

    return 0;
}