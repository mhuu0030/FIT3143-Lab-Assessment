/**
 * FIT3143 Parallel Computing - Lab 1, Task 1
 * Serial code to find prime numbers strictly less than an integer n.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/**
 * Checks if a given integer is a prime number.
 * Utilizes the square root optimization to eliminate unnecessary computations.
 * * @param k The integer to check for primality.
 * @return true if k is prime, false otherwise.
 */
bool is_prime(int k) {
    // 0 and 1 are not prime numbers
    if (k <= 1) return false;
    
    // 2 is the only even prime number
    if (k == 2) return true;
    
    // Eliminate all other even numbers immediately
    if (k % 2 == 0) return false;

    // Optimization: We only need to check for factors up to the square root of k.
    // If m * n = k, and m > sqrt(k), then n must be < sqrt(k).
    int limit = (int)sqrt((double)k);
    
    // Check odd divisors from 3 up to the square root of k
    for (int i = 3; i <= limit; i += 2) {
        if (k % i == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    int n;

    // Prompt user for input
    printf("Enter an integer n (program will find primes strictly less than n): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Error: Please enter a valid positive integer.\n");
        return 1;
    }

    FILE *file = NULL;
    
    // Determine output destination based on the size of n
    if (n >= 100) {
        file = fopen("primes.txt", "w");
        if (file == NULL) {
            printf("Error: Could not open primes.txt for writing.\n");
            return 1;
        }
        printf("Calculating... Output will be written to primes.txt\n");
    } else {
        printf("Prime numbers strictly less than %d are:\n", n);
    }

    // Start time measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Search for prime numbers strictly less than n
    // Checking from 2 incrementally naturally sorts the output in ascending order
    for (int i = 2; i < n; i++) {
        if (is_prime(i)) {
            if (n < 100) {
                // Standard output for small n values
                printf("%d ", i);
            } else {
                // Text file output for larger n values
                fprintf(file, "%d\n", i);
            }
        }
    }

    // End time measurement
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Close the file if it was opened
    if (file != NULL) {
        fclose(file);
    } else {
        printf("\n"); // Add a newline if we printed to terminal
    }

    // Calculate execution time in seconds
    double time_taken = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Print the execution time
    printf("Execution time: %f seconds\n", time_taken);

    return 0;
}