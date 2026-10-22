#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NEIGHBOURS 4
#define REQUIRED_WORLD_SIZE 32
#define GRID_CASE_COUNT 30
#define LINE_CASE_COUNT 30
#define MESSAGE_TYPE_COUNT 9

/*
 * FIT3143 Lab 3 - Task 3 communication analysis
 *
 * The program replays the message payloads and communication paths used by
 * Tasks 1 and 2. It deliberately measures round-trip time and divides by two
 * for base-station traffic. This avoids assuming that MPI_Wtime clocks on
 * different CAAS compute nodes are synchronised.
 *
 * Usage:
 *   task3_communication_analysis [iterations] [warmups] [output_dir]
 *                                [allow_partial]
 *
 * A complete HD data run requires 32 MPI processes: one base station and up
 * to 31 EV nodes. allow_partial=1 is only for local smoke tests.
 */

typedef struct {
    int rank;
    int free_ports;
    int cycle;
} StatusUpdate;

typedef struct {
    int node_rank;
    int row;
    int col;
    int cycle;
    int total_ports;
    int free_ports;
    int threshold;
    int neighbour_count;
    int neighbour_rank[MAX_NEIGHBOURS];
    int neighbour_free[MAX_NEIGHBOURS];
    int neighbour_cycle[MAX_NEIGHBOURS];
    int neighbour_messages;
    int neighbour_bytes;
    long long timestamp_ms;
    double neighbour_exchange_seconds;
} CongestionReport;

typedef struct {
    int candidate_count;
    int candidate_ranks[MAX_NEIGHBOURS];
} Task1RedirectReply;

typedef struct {
    const char *name;
    int bytes;
    int neighbour_path;
} MessageType;

typedef struct {
    double sum;
    double sum_sq;
    double minimum;
    double maximum;
    long long samples;
} Stats;

typedef struct {
    int rows;
    int cols;
} GridCase;

static const GridCase GRID_CASES[GRID_CASE_COUNT] = {
    {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6},
    {2, 7}, {2, 8}, {2, 9}, {2, 10}, {2, 11},
    {2, 12}, {2, 13}, {2, 14}, {2, 15},
    {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6},
    {3, 7}, {3, 8}, {3, 9}, {3, 10},
    {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
    {4, 7}, {5, 2}
};

static const MessageType MESSAGE_TYPES[MESSAGE_TYPE_COUNT] = {
    {"neighbour_request", (int)sizeof(int), 1},
    {"neighbour_response", (int)(2 * sizeof(int)), 1},
    {"node_status_update", (int)sizeof(StatusUpdate), 0},
    {"congestion_report", (int)sizeof(CongestionReport), 0},
    {"task1_redirect_reply", (int)sizeof(Task1RedirectReply), 0},
    {"task2_redirect_count", (int)sizeof(int), 0},
    {"task2_redirect_list", (int)(MAX_NEIGHBOURS * sizeof(int)), 0},
    {"done", (int)sizeof(int), 0},
    {"terminate", (int)sizeof(int), 0}
};

static void fail(const char *message, int world_rank)
{
    if (world_rank == 0)
        fprintf(stderr, "ERROR: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

/*
 * Measure local elapsed time with an operating-system monotonic clock.
 * MPI_Wtime is normally suitable, but some virtualised WSL/MPICH setups can
 * expose non-synchronised CPU timing sources when a process migrates between
 * virtual cores. CLOCK_MONOTONIC guarantees that a local interval cannot run
 * backwards. Cross-process clock synchronisation is still not assumed.
 */
static double monotonic_seconds(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static int parse_positive(const char *text, const char *name, int rank)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);

    if (text[0] == '\0' || end == NULL || *end != '\0' ||
        value <= 0 || value > 1000000L) {
        if (rank == 0)
            fprintf(stderr, "Invalid %s: %s\n", name, text);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return (int)value;
}

static Stats empty_stats(void)
{
    Stats result;
    result.sum = 0.0;
    result.sum_sq = 0.0;
    result.minimum = DBL_MAX;
    result.maximum = 0.0;
    result.samples = 0;
    return result;
}

static void add_sample(Stats *stats, double seconds)
{
    stats->sum += seconds;
    stats->sum_sq += seconds * seconds;
    if (seconds < stats->minimum)
        stats->minimum = seconds;
    if (seconds > stats->maximum)
        stats->maximum = seconds;
    stats->samples++;
}

static Stats reduce_stats(Stats local, MPI_Comm communicator)
{
    Stats global = empty_stats();

    MPI_Reduce(&local.sum, &global.sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, communicator);
    MPI_Reduce(&local.sum_sq, &global.sum_sq, 1, MPI_DOUBLE,
               MPI_SUM, 0, communicator);
    MPI_Reduce(&local.minimum, &global.minimum, 1, MPI_DOUBLE,
               MPI_MIN, 0, communicator);
    MPI_Reduce(&local.maximum, &global.maximum, 1, MPI_DOUBLE,
               MPI_MAX, 0, communicator);
    MPI_Reduce(&local.samples, &global.samples, 1, MPI_LONG_LONG,
               MPI_SUM, 0, communicator);

    return global;
}

/*
 * Measure one directed message to every Cartesian neighbour. The four
 * Sendrecv phases match safely in both dimensions and run concurrently on
 * all EV nodes. Each sample is a node's communication-phase time divided by
 * the number of real messages it sent in that phase.
 */
static Stats benchmark_neighbours(
    MPI_Comm cart,
    int payload_bytes,
    int iterations,
    int warmups,
    int tag
) {
    Stats stats = empty_stats();
    char *send_buffer = calloc((size_t)payload_bytes, 1);
    char *recv_buffer = calloc((size_t)payload_bytes, 1);
    int up, down, left, right;
    int neighbours;

    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Unable to allocate benchmark buffers\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);
    neighbours = (up != MPI_PROC_NULL) + (down != MPI_PROC_NULL) +
                 (left != MPI_PROC_NULL) + (right != MPI_PROC_NULL);

    for (int round = -warmups; round < iterations; round++) {
        MPI_Barrier(cart);
        double start = monotonic_seconds();

        MPI_Sendrecv(send_buffer, payload_bytes, MPI_BYTE, down, tag,
                     recv_buffer, payload_bytes, MPI_BYTE, up, tag,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer, payload_bytes, MPI_BYTE, up, tag + 1,
                     recv_buffer, payload_bytes, MPI_BYTE, down, tag + 1,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer, payload_bytes, MPI_BYTE, right, tag + 2,
                     recv_buffer, payload_bytes, MPI_BYTE, left, tag + 2,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer, payload_bytes, MPI_BYTE, left, tag + 3,
                     recv_buffer, payload_bytes, MPI_BYTE, right, tag + 3,
                     cart, MPI_STATUS_IGNORE);

        double elapsed = monotonic_seconds() - start;
        if (round >= 0 && neighbours > 0)
            add_sample(&stats, elapsed / (double)neighbours);
    }

    free(send_buffer);
    free(recv_buffer);
    return stats;
}

/*
 * All EV nodes concurrently send the selected Task 1/2 payload to the base
 * station. The base echoes the same payload. Each EV node measures RTT/2,
 * which is a defensible one-way-delay estimate without synchronised clocks.
 */
static Stats benchmark_base_station(
    MPI_Comm active,
    int payload_bytes,
    int iterations,
    int warmups,
    int tag
) {
    Stats stats = empty_stats();
    int rank, size;
    char *buffer = calloc((size_t)payload_bytes, 1);

    MPI_Comm_rank(active, &rank);
    MPI_Comm_size(active, &size);
    if (buffer == NULL) {
        fprintf(stderr, "Unable to allocate benchmark buffer\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int round = -warmups; round < iterations; round++) {
        MPI_Barrier(active);

        if (rank == 0) {
            for (int node = 1; node < size; node++) {
                MPI_Status status;
                MPI_Recv(buffer, payload_bytes, MPI_BYTE, MPI_ANY_SOURCE,
                         tag, active, &status);
                MPI_Send(buffer, payload_bytes, MPI_BYTE, status.MPI_SOURCE,
                         tag + 1, active);
            }
        } else {
            double start = monotonic_seconds();
            MPI_Send(buffer, payload_bytes, MPI_BYTE, 0, tag, active);
            MPI_Recv(buffer, payload_bytes, MPI_BYTE, 0, tag + 1,
                     active, MPI_STATUS_IGNORE);
            double one_way = (monotonic_seconds() - start) / 2.0;
            if (round >= 0)
                add_sample(&stats, one_way);
        }
    }

    free(buffer);
    return stats;
}

static void write_header(FILE *file)
{
    fprintf(file,
        "topology,case_index,rows,cols,node_count,message_type,path,"
        "payload_bytes,iterations,samples,mean_us,min_us,max_us,stddev_us\n");
}

static void write_result(
    FILE *file,
    const char *topology,
    int case_index,
    int rows,
    int cols,
    MessageType message,
    int iterations,
    Stats stats
) {
    double mean = stats.samples > 0
        ? stats.sum / (double)stats.samples : 0.0;
    double variance = stats.samples > 0
        ? stats.sum_sq / (double)stats.samples - mean * mean : 0.0;
    if (variance < 0.0)
        variance = 0.0;
    double stddev = sqrt(variance);
    double minimum = stats.samples > 0 ? stats.minimum : 0.0;

    fprintf(file,
        "%s,%d,%d,%d,%d,%s,%s,%d,%d,%lld,%.6f,%.6f,%.6f,%.6f\n",
        topology, case_index, rows, cols, rows * cols,
        message.name,
        message.neighbour_path ? "neighbour" : "base_station",
        message.bytes, iterations, stats.samples,
        mean * 1.0e6, minimum * 1.0e6,
        stats.maximum * 1.0e6, stddev * 1.0e6);
    fflush(file);
}

static void run_case(
    const char *topology,
    int case_index,
    int rows,
    int cols,
    int iterations,
    int warmups,
    FILE *output
) {
    int world_rank;
    int node_count = rows * cols;
    int active_colour;
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm cart = MPI_COMM_NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    active_colour = world_rank <= node_count ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, active_colour, world_rank, &active);

    if (active != MPI_COMM_NULL) {
        int active_rank;
        MPI_Comm_rank(active, &active_rank);
        MPI_Comm_split(active, active_rank == 0 ? MPI_UNDEFINED : 0,
                       active_rank, &node_comm);

        if (node_comm != MPI_COMM_NULL) {
            int dimensions[2] = {rows, cols};
            int periods[2] = {0, 0};
            MPI_Cart_create(node_comm, 2, dimensions, periods, 0, &cart);
        }

        for (int message_index = 0;
             message_index < MESSAGE_TYPE_COUNT;
             message_index++) {
            MessageType message = MESSAGE_TYPES[message_index];
            Stats local = empty_stats();
            int tag = 1000 + message_index * 10;

            if (message.neighbour_path) {
                if (cart != MPI_COMM_NULL) {
                    local = benchmark_neighbours(
                        cart, message.bytes, iterations, warmups, tag);
                }
            } else {
                local = benchmark_base_station(
                    active, message.bytes, iterations, warmups, tag);
            }

            Stats global = reduce_stats(local, active);
            if (active_rank == 0) {
                write_result(output, topology, case_index, rows, cols,
                             message, iterations, global);
            }
        }

        if (cart != MPI_COMM_NULL)
            MPI_Comm_free(&cart);
        if (node_comm != MPI_COMM_NULL)
            MPI_Comm_free(&node_comm);
        MPI_Comm_free(&active);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

static void write_metadata(
    const char *output_dir,
    int world_size,
    int iterations,
    int warmups,
    int grid_cases,
    int line_cases
) {
    char path[1024];
    char library[MPI_MAX_LIBRARY_VERSION_STRING];
    int library_length = 0;
    snprintf(path, sizeof(path), "%s/task3_run_metadata.txt", output_dir);
    FILE *file = fopen(path, "w");
    if (file == NULL)
        return;

    time_t now = time(NULL);
    fprintf(file, "Task 3 communication-analysis run\n");
    fprintf(file, "UTC timestamp: %s", asctime(gmtime(&now)));
    fprintf(file, "MPI processes: %d\n", world_size);
    fprintf(file, "EV-node capacity: %d\n", world_size - 1);
    fprintf(file, "Measured iterations per message/case: %d\n", iterations);
    fprintf(file, "Warm-up iterations per message/case: %d\n", warmups);
    fprintf(file, "Grid cases completed: %d/%d\n",
            grid_cases, GRID_CASE_COUNT);
    fprintf(file, "Line cases completed: %d/%d\n",
            line_cases, LINE_CASE_COUNT);
    fprintf(file, "Delay method: neighbour phase per directed message; "
                  "base-station RTT/2\n");
    MPI_Get_library_version(library, &library_length);
    fprintf(file, "MPI library: %.*s\n", library_length, library);
    fclose(file);
}

int main(int argc, char **argv)
{
    int world_rank, world_size;
    int iterations = 100;
    int warmups = 20;
    int allow_partial = 0;
    const char *output_dir = ".";
    FILE *grid_output = NULL;
    FILE *line_output = NULL;
    int grid_completed = 0;
    int line_completed = 0;
    char grid_path[1024];
    char line_path[1024];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 1)
        iterations = parse_positive(argv[1], "iterations", world_rank);
    if (argc > 2)
        warmups = parse_positive(argv[2], "warmups", world_rank);
    if (argc > 3)
        output_dir = argv[3];
    if (argc > 4)
        allow_partial = atoi(argv[4]) != 0;

    if (world_size < REQUIRED_WORLD_SIZE && !allow_partial) {
        fail("complete analysis requires 32 MPI processes; use "
             "allow_partial=1 only for a local smoke test", world_rank);
    }

    if (world_rank == 0) {
        snprintf(grid_path, sizeof(grid_path),
                 "%s/task3_grid_results.csv", output_dir);
        snprintf(line_path, sizeof(line_path),
                 "%s/task3_line_results.csv", output_dir);
        grid_output = fopen(grid_path, "w");
        line_output = fopen(line_path, "w");
        if (grid_output == NULL || line_output == NULL)
            fail("cannot create Task 3 CSV output files", world_rank);
        write_header(grid_output);
        write_header(line_output);
    }

    for (int i = 0; i < GRID_CASE_COUNT; i++) {
        int nodes = GRID_CASES[i].rows * GRID_CASES[i].cols;
        if (nodes + 1 <= world_size) {
            run_case("grid", i + 1, GRID_CASES[i].rows,
                     GRID_CASES[i].cols, iterations, warmups, grid_output);
            grid_completed++;
        }
    }

    for (int i = 0; i < LINE_CASE_COUNT; i++) {
        int nodes = i + 2;
        if (nodes + 1 <= world_size) {
            run_case("line", i + 1, 1, nodes,
                     iterations, warmups, line_output);
            line_completed++;
        }
    }

    if (world_rank == 0) {
        fclose(grid_output);
        fclose(line_output);
        write_metadata(output_dir, world_size, iterations, warmups,
                       grid_completed, line_completed);

        printf("Task 3 analysis complete: %d grid cases, %d line cases, "
               "%d message types.\n",
               grid_completed, line_completed, MESSAGE_TYPE_COUNT);
        printf("Grid CSV: %s\nLine CSV: %s\n", grid_path, line_path);

        if (!allow_partial &&
            (grid_completed != GRID_CASE_COUNT ||
             line_completed != LINE_CASE_COUNT)) {
            fail("complete run did not produce all required cases",
                 world_rank);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
