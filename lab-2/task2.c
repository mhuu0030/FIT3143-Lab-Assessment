/**
 * FIT3143 Parallel Computing - Lab 1, Task 2
 * Parallel code to find prime numbers strictly less than an integer n
 * using POSIX Threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define MAX_THREADS 64

int n;
int num_threads;
bool *prime_results = NULL;
double threadTime[MAX_THREADS];

// POSIX Thread function prototype
void *ThreadFunc(void *pArg);

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

    // Prompt user for input
    printf("Enter an integer n (program will find primes strictly less than n): ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Error: Please enter a valid positive integer.\n");
        return 1;
    }

    // Prompt user for number of threads
    printf("Enter the number of threads: ");
    if (scanf("%d", &num_threads) != 1 ||
        num_threads <= 0 ||
        num_threads > MAX_THREADS) {
        printf("Error: Please enter a number of threads between 1 and %d.\n",
               MAX_THREADS);
        return 1;
    }

    // Time measurements
    struct timespec start_time, end_time;
    struct timespec start_comp, end_comp;

    // Start total time measurement
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    FILE *file = NULL;

    // Determine output destination based on the size of n
    if (n >= 100) {
        file = fopen("task2primes.txt", "w");

        if (file == NULL) {
            printf("Error: Could not open task2primes.txt for writing.\n");
            return 1;
        }

        printf("Calculating... Output will be written to task2primes.txt\n");
    } else {
        printf("Prime numbers strictly less than %d are:\n", n);
    }

    // Store the result for each number
    prime_results = (bool *)calloc(n, sizeof(bool));

    if (prime_results == NULL) {
        printf("Error: Memory allocation failed.\n");

        if (file != NULL) {
            fclose(file);
        }

        return 1;
    }

    pthread_t tid[MAX_THREADS];
    int threadNum[MAX_THREADS];

    // Start computational time measurement
    clock_gettime(CLOCK_MONOTONIC, &start_comp);

    // 2 is the only even prime number, so handle it separately
    if (n > 2) {
        prime_results[2] = true;
    }

    // Fork - create the threads
    for (int i = 0; i < num_threads; i++) {
        threadNum[i] = i;
        pthread_create(&tid[i], 0, ThreadFunc, &threadNum[i]);
    }

    // Join - wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }

    // End computational time measurement
    clock_gettime(CLOCK_MONOTONIC, &end_comp);

    // Output prime numbers in ascending order
    for (int i = 2; i < n; i++) {
        if (prime_results[i]) {
            if (n < 100) {
                // Standard output for small n values
                printf("%d ", i);
            } else {
                // Text file output for larger n values
                fprintf(file, "%d\n", i);
            }
        }
    }

    // Close the file if it was opened
    if (file != NULL) {
        fclose(file);
    } else {
        printf("\n"); // Add a newline if we printed to terminal
    }

    free(prime_results);

    // End total time measurement
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate computational time in seconds
    double comp_time = (end_comp.tv_sec - start_comp.tv_sec) +
                       (end_comp.tv_nsec - start_comp.tv_nsec) / 1e9;

    // Calculate total execution time in seconds
    double time_taken = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Print time taken by each thread
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %d CPU time: %f seconds\n",
               i, threadTime[i]);
    }

    // Print execution times
    printf("Computational time: %f seconds\n", comp_time);
    printf("Total execution time: %f seconds\n", time_taken);

    return 0;
}


// POSIX Thread function
void *ThreadFunc(void *pArg) {

    int my_rank = *((int *)pArg);

    struct timespec threadStart, threadEnd;
    double thread_time;

    // Start measuring time taken by this thread
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadStart);

    /*
     * Cyclic workload distribution using odd numbers only.
     *
     * For 4 threads:
     *
     * Thread 0: 3, 11, 19, 27, ...
     * Thread 1: 5, 13, 21, 29, ...
     * Thread 2: 7, 15, 23, 31, ...
     * Thread 3: 9, 17, 25, 33, ...
     */

    int first = 3 + (2 * my_rank);
    int step = 2 * num_threads;

    for (int i = first; i < n; i += step) {
        prime_results[i] = is_prime(i);
    }

    // End measuring time taken by this thread
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &threadEnd);

    thread_time = (threadEnd.tv_sec - threadStart.tv_sec) * 1e9;
    thread_time = (thread_time +
                  (threadEnd.tv_nsec - threadStart.tv_nsec)) * 1e-9;

    threadTime[my_rank] = thread_time;

    return NULL;
}