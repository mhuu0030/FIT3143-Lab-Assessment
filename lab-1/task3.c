/**
 * FIT3143 Parallel Computing - Lab 1, Task 3
 * OpenMP parallel code to find prime numbers strictly less than an integer n.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Include OpenMP library

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization to eliminate unnecessary computations.
 * * @param k The integer to check for primality.
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
    // This allows threads to flag primes independently without locking or printing out of order.
    bool *prime_flags = (bool *)calloc(n, sizeof(bool));
    if (prime_flags == NULL) {
        printf("Error: Memory allocation failed.\n");
        if (file) fclose(file);
        return 1;
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // OPENMP PARALLEL REGION
    // We use dynamic scheduling because checking larger numbers takes slightly longer.
    // Dynamic scheduling ensures a balanced workload distribution among threads.
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 2; i < n; i++) {
        if (is_prime(i)) {
            prime_flags[i] = true; // Mark as prime
        }
    }

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

    double time_taken = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("Execution time: %f seconds\n", time_taken);

    return 0;
}