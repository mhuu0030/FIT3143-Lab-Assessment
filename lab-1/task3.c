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

    printf("Enter an integer n (program will find primes strictly less than n): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Error: Please enter a valid positive integer.\n");
        return 1;
    }

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

    // Allocate a boolean array to keep track of primes. 
    bool *prime_flags = (bool *)calloc(n, sizeof(bool));
    if (prime_flags == NULL) {
        printf("Error: Memory allocation failed.\n");
        if (file) fclose(file);
        return 1;
    }

    // Measure overall wall-clock time
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // 2 is the only even prime number, so handle it sequentially before opening threads
    if (n > 2) {
        prime_flags[2] = true;
    }

    // OPENMP PARALLEL REGION
    #pragma omp parallel 
    {
        // Each thread gets its own thread ID and local timing variables
        int tid = omp_get_thread_num();
        struct timespec threadStart, threadEnd;
        double threadTime;

        // Start measuring CPU time used by this specific thread
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadStart);

        // Work-sharing construct: dynamically distribute the loop iterations
        // OPTIMIZATION: Start at 3 and increment by 2 to skip all even numbers
        #pragma omp for schedule(dynamic, 128)
        for (int i = 3; i < n; i += 2) {
            if (is_prime(i)) {
                prime_flags[i] = true; // Mark as prime
            }
        }

        // End measuring CPU time for this thread
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadEnd);
        
        // Calculate per-thread execution time
        threadTime = (threadEnd.tv_sec - threadStart.tv_sec) + 
                     (threadEnd.tv_nsec - threadStart.tv_nsec) / 1e9;

        // Print the individual thread's CPU time safely
        #pragma omp critical
        {
            printf("Thread %d CPU time (s): %f\n", tid, threadTime);
        }
    } // End of parallel region

    clock_gettime(CLOCK_MONOTONIC, &end_time);

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

    free(prime_flags); // Clean up memory

    // Calculate total overall wall-clock execution time
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("\nOverall Execution time: %f seconds\n", time_taken);

    return 0;
}