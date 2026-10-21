#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <float.h>

#define MAX_NEIGHBOURS 4
#define DEFAULT_PORTS 6
#define DEFAULT_CYCLES 8
#define DEFAULT_INTERVAL_MS 1000
#define DEFAULT_THRESHOLD 1

/*
 * FIT3143 Lab 3 - Task 1
 * BLOCKING MPI BASELINE
 *
 * Reference implementation derived from lab3_task1.c.
 *
 * MPI_COMM_WORLD rank 0:
 *     Base-station test harness.
 *
 * MPI_COMM_WORLD ranks 1..N:
 *     EV charging nodes.
 *
 * EV nodes are placed in a non-periodic 2-D Cartesian topology.
 * Cartesian reordering is disabled, so:
 *
 *     world rank = cart rank + 1
 *
 * OpenMP is still used to simulate charging ports, while MPI
 * communication is performed by the main thread.
 *
 * IMPORTANT DIFFERENCE FROM THE NON-BLOCKING VERSION:
 *
 * The original lab3_task1.c uses MPI_Irecv, MPI_Isend, MPI_Testall
 * and MPI_Waitall for event-driven neighbour communication.
 *
 * This baseline deliberately uses blocking MPI_Sendrecv calls.
 * Every EV node participates in a synchronous neighbour-status
 * exchange every cycle. Only congested nodes USE those results to
 * redirect locally or report congestion to the base station.
 *
 * This makes the implementation safe from blocking-send deadlock
 * while also giving a clear blocking baseline for comparison with
 * the non-blocking implementation.
 */

enum {
    TAG_NEIGHBOUR_VERTICAL_1   = 100,
    TAG_NEIGHBOUR_VERTICAL_2   = 101,
    TAG_NEIGHBOUR_HORIZONTAL_1 = 102,
    TAG_NEIGHBOUR_HORIZONTAL_2 = 103,

    TAG_CONGESTION_REPORT = 200,
    TAG_REDIRECT          = 201,

    TAG_DONE      = 998,
    TAG_TERMINATE = 999
};

enum {
    DIR_TOP = 0,
    DIR_BOTTOM = 1,
    DIR_LEFT = 2,
    DIR_RIGHT = 3
};

typedef struct {
    int node_rank;
    int coord_row;
    int coord_col;

    int total_ports;
    int local_available;
    int threshold;

    int valid_neighbour_count;
    int neighbour_ranks[MAX_NEIGHBOURS];
    int neighbour_available[MAX_NEIGHBOURS];

    int neighbour_messages;
    int neighbour_bytes;

    long long epoch_ms;
    double neighbour_exchange_seconds;
} CongestionReport;

typedef struct {
    int candidate_count;
    int candidate_ranks[MAX_NEIGHBOURS];
} RedirectReply;

typedef struct {
    int rows;
    int cols;
    int ports;
    int cycles;
    int interval_ms;
    int threshold;
    int force_low;
} Config;

/* =========================================================
 * COMMON UTILITIES
 * ========================================================= */

static long long now_epoch_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_REALTIME,
        &ts
    );

    return (long long)ts.tv_sec * 1000LL
         + ts.tv_nsec / 1000000LL;
}

static void format_time(
    long long epoch_ms,
    char *buffer,
    size_t size
) {
    time_t sec =
        (time_t)(epoch_ms / 1000LL);

    struct tm t;
    char base[64];

    localtime_r(
        &sec,
        &t
    );

    strftime(
        base,
        sizeof(base),
        "%Y-%m-%d %H:%M:%S",
        &t
    );

    snprintf(
        buffer,
        size,
        "%s.%03lld",
        base,
        epoch_ms % 1000LL
    );
}

static void sleep_ms(int ms)
{
    if (ms <= 0)
        return;

    struct timespec req;

    req.tv_sec = ms / 1000;
    req.tv_nsec =
        (long)(ms % 1000) * 1000000L;

    while (nanosleep(&req, &req) != 0) {
        /* Retry interrupted sleep. */
    }
}

/* =========================================================
 * OPENMP CHARGING-PORT SIMULATION
 * ========================================================= */

/*
 * Shared port-status array:
 *
 * 1 = free
 * 0 = occupied
 */
static void update_ports(
    int *port_status,
    unsigned int *seeds,
    int ports,
    int force_low,
    int cycle
) {
    int free_probability =
        force_low ? 10 : 55;

    #pragma omp parallel for schedule(static)
    for (int i = 0;
         i < ports;
         i++) {

        if (force_low &&
            cycle == 0) {

            /*
             * Deterministic cycle-0 congestion
             * for stress testing.
             */
            port_status[i] = 0;

        } else {

            int value =
                (int)(rand_r(&seeds[i]) %
                      100U);

            port_status[i] =
                (value < free_probability)
                ? 1
                : 0;
        }
    }
}

static int count_available(
    const int *port_status,
    int ports
) {
    int total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0;
         i < ports;
         i++) {

        total += port_status[i];
    }

    return total;
}

/* =========================================================
 * CARTESIAN TOPOLOGY HELPERS
 * ========================================================= */

static int cart_to_world(
    int cart_rank
) {
    if (cart_rank ==
        MPI_PROC_NULL) {

        return MPI_PROC_NULL;
    }

    return cart_rank + 1;
}

/*
 * Fixed neighbour order:
 *
 * [0] top
 * [1] bottom
 * [2] left
 * [3] right
 */
static int discover_neighbours(
    MPI_Comm cart_comm,
    int cart_neighbours[MAX_NEIGHBOURS],
    int world_neighbours[MAX_NEIGHBOURS]
) {
    MPI_Cart_shift(
        cart_comm,
        0,
        1,
        &cart_neighbours[DIR_TOP],
        &cart_neighbours[DIR_BOTTOM]
    );

    MPI_Cart_shift(
        cart_comm,
        1,
        1,
        &cart_neighbours[DIR_LEFT],
        &cart_neighbours[DIR_RIGHT]
    );

    int count = 0;

    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        world_neighbours[i] =
            cart_to_world(
                cart_neighbours[i]
            );

        if (cart_neighbours[i] !=
            MPI_PROC_NULL) {

            count++;
        }
    }

    return count;
}

/* =========================================================
 * BLOCKING NEIGHBOUR COMMUNICATION
 * ========================================================= */

/*
 * Safely exchange the current free-port count with all
 * four Cartesian neighbours using blocking MPI_Sendrecv.
 *
 * Why MPI_Sendrecv?
 * -----------------
 * A naive sequence such as:
 *
 *     MPI_Send(...)
 *     MPI_Recv(...)
 *
 * can deadlock when neighbouring processes both block while
 * trying to send to one another.
 *
 * MPI_Sendrecv performs the matching blocking send and receive
 * as one MPI operation.
 *
 * Four directional phases are used:
 *
 * 1. send to TOP,    receive from BOTTOM
 * 2. send to BOTTOM, receive from TOP
 * 3. send to LEFT,   receive from RIGHT
 * 4. send to RIGHT,  receive from LEFT
 *
 * MPI_PROC_NULL is valid in MPI_Sendrecv, so boundary nodes do
 * not need special communication branches.
 */
static double blocking_neighbour_exchange(
    MPI_Comm cart_comm,
    int current_available,
    const int cart_neighbours[MAX_NEIGHBOURS],
    int neighbour_available[MAX_NEIGHBOURS]
) {
    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        neighbour_available[i] = -1;
    }

    double start =
        MPI_Wtime();

    /*
     * Send current availability upward.
     * Receive lower neighbour's availability.
     */
    MPI_Sendrecv(
        &current_available,
        1,
        MPI_INT,
        cart_neighbours[DIR_TOP],
        TAG_NEIGHBOUR_VERTICAL_1,

        &neighbour_available[DIR_BOTTOM],
        1,
        MPI_INT,
        cart_neighbours[DIR_BOTTOM],
        TAG_NEIGHBOUR_VERTICAL_1,

        cart_comm,
        MPI_STATUS_IGNORE
    );

    /*
     * Send downward.
     * Receive top neighbour.
     */
    MPI_Sendrecv(
        &current_available,
        1,
        MPI_INT,
        cart_neighbours[DIR_BOTTOM],
        TAG_NEIGHBOUR_VERTICAL_2,

        &neighbour_available[DIR_TOP],
        1,
        MPI_INT,
        cart_neighbours[DIR_TOP],
        TAG_NEIGHBOUR_VERTICAL_2,

        cart_comm,
        MPI_STATUS_IGNORE
    );

    /*
     * Send left.
     * Receive right neighbour.
     */
    MPI_Sendrecv(
        &current_available,
        1,
        MPI_INT,
        cart_neighbours[DIR_LEFT],
        TAG_NEIGHBOUR_HORIZONTAL_1,

        &neighbour_available[DIR_RIGHT],
        1,
        MPI_INT,
        cart_neighbours[DIR_RIGHT],
        TAG_NEIGHBOUR_HORIZONTAL_1,

        cart_comm,
        MPI_STATUS_IGNORE
    );

    /*
     * Send right.
     * Receive left neighbour.
     */
    MPI_Sendrecv(
        &current_available,
        1,
        MPI_INT,
        cart_neighbours[DIR_RIGHT],
        TAG_NEIGHBOUR_HORIZONTAL_2,

        &neighbour_available[DIR_LEFT],
        1,
        MPI_INT,
        cart_neighbours[DIR_LEFT],
        TAG_NEIGHBOUR_HORIZONTAL_2,

        cart_comm,
        MPI_STATUS_IGNORE
    );

    return MPI_Wtime() - start;
}

/* =========================================================
 * TASK 1 BASE-STATION TEST HARNESS
 * ========================================================= */

/*
 * This is intentionally NOT Task 2.
 *
 * It exists only so Task 1 can send congestion reports
 * and receive a reply.
 *
 * This blocking baseline also uses blocking MPI_Probe,
 * MPI_Recv and MPI_Send at the base station.
 */
static void run_base_station(
    int workers
) {
    FILE *log =
        fopen(
            "task1_blocking_base_station.log",
            "w"
        );

    if (!log) {

        fprintf(
            stderr,
            "Base station: log open failed\n"
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            1
        );
    }

    int done_count = 0;
    int alerts = 0;

    long long report_bytes = 0;

    double start =
        MPI_Wtime();

    fprintf(
        log,
        "TASK 1 BLOCKING MPI BASELINE\n"
        "EV nodes: %d\n\n",
        workers
    );

    fflush(log);

    while (done_count <
           workers) {

        MPI_Status status;

        /*
         * Blocking wait for the next incoming
         * Task-1 message.
         */
        MPI_Probe(
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &status
        );

        if (status.MPI_TAG ==
            TAG_CONGESTION_REPORT) {

            CongestionReport report;

            memset(
                &report,
                0,
                sizeof(report)
            );

            MPI_Recv(
                &report,
                (int)sizeof(report),
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_CONGESTION_REPORT,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            alerts++;

            report_bytes +=
                sizeof(report);

            char timestamp[80];

            format_time(
                report.epoch_ms,
                timestamp,
                sizeof(timestamp)
            );

            fprintf(
                log,
                "----------------------------------------\n"
                "[%s] CONGESTION ALERT\n"
                "Node: %d\n"
                "Coordinates: (%d,%d)\n"
                "Available ports: %d/%d\n"
                "Threshold: %d\n"
                "Neighbour messages: %d\n"
                "Neighbour payload bytes: %d\n"
                "Neighbour exchange time: %.9f s\n",
                timestamp,
                report.node_rank,
                report.coord_row,
                report.coord_col,
                report.local_available,
                report.total_ports,
                report.threshold,
                report.neighbour_messages,
                report.neighbour_bytes,
                report.neighbour_exchange_seconds
            );

            for (int i = 0;
                 i < MAX_NEIGHBOURS;
                 i++) {

                if (report.neighbour_ranks[i] !=
                    MPI_PROC_NULL) {

                    fprintf(
                        log,
                        "Neighbour %d: %d available\n",
                        report.neighbour_ranks[i],
                        report.neighbour_available[i]
                    );
                }
            }

            /*
             * Placeholder Task-1 reply.
             * Task 2 replaces this with the actual
             * nearest-available-node computation.
             */
            RedirectReply reply;

            memset(
                &reply,
                0,
                sizeof(reply)
            );

            for (int i = 0;
                 i < MAX_NEIGHBOURS;
                 i++) {

                reply.candidate_ranks[i] = -1;
            }

            MPI_Send(
                &reply,
                (int)sizeof(reply),
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_REDIRECT,
                MPI_COMM_WORLD
            );

            fprintf(
                log,
                "Placeholder response sent.\n\n"
            );

            fflush(log);

        } else if (status.MPI_TAG ==
                   TAG_DONE) {

            int value = 0;

            MPI_Recv(
                &value,
                1,
                MPI_INT,
                status.MPI_SOURCE,
                TAG_DONE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            done_count++;

            fprintf(
                log,
                "Node %d completed (%d/%d)\n",
                status.MPI_SOURCE,
                done_count,
                workers
            );

            fflush(log);
        }
    }

    /*
     * Gracefully terminate every EV node only after
     * all of them have completed their scheduled cycles.
     */
    for (int rank = 1;
         rank <= workers;
         rank++) {

        int terminate = 1;

        MPI_Send(
            &terminate,
            1,
            MPI_INT,
            rank,
            TAG_TERMINATE,
            MPI_COMM_WORLD
        );
    }

    double total =
        MPI_Wtime() - start;

    fprintf(
        log,
        "\n========================================\n"
        "SIMULATION SUMMARY\n"
        "========================================\n"
        "Implementation: BLOCKING MPI\n"
        "Simulation time: %.9f s\n"
        "Congestion alerts: %d\n"
        "Alert payload bytes: %lld\n"
        "Termination messages: %d\n",
        total,
        alerts,
        report_bytes,
        workers
    );

    fclose(log);

    printf(
        "[BASE BLOCKING] "
        "Completed. Alerts=%d Time=%.3f s\n",
        alerts,
        total
    );

    fflush(stdout);
}

/* =========================================================
 * TASK 1 EV CHARGING NODE - BLOCKING MPI VERSION
 * ========================================================= */

static void run_charging_node(
    int world_rank,
    MPI_Comm worker_comm,
    MPI_Comm cart_comm,
    const Config *cfg
) {
    int cart_rank;
    int coords[2];

    MPI_Comm_rank(
        cart_comm,
        &cart_rank
    );

    MPI_Cart_coords(
        cart_comm,
        cart_rank,
        2,
        coords
    );

    int cart_neighbours[MAX_NEIGHBOURS];
    int world_neighbours[MAX_NEIGHBOURS];

    int neighbour_count =
        discover_neighbours(
            cart_comm,
            cart_neighbours,
            world_neighbours
        );

    /*
     * Shared OpenMP charging-port arrays.
     */
    int *port_status =
        malloc(
            (size_t)cfg->ports *
            sizeof(int)
        );

    unsigned int *seeds =
        malloc(
            (size_t)cfg->ports *
            sizeof(unsigned int)
        );

    if (!port_status ||
        !seeds) {

        fprintf(
            stderr,
            "Node %d: allocation failed\n",
            world_rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            2
        );
    }

    for (int i = 0;
         i < cfg->ports;
         i++) {

        port_status[i] = 1;

        seeds[i] =
            (unsigned int)time(NULL)
            ^ ((unsigned int)world_rank *
               1009U)
            ^ ((unsigned int)(i + 1) *
               7919U);
    }

    char filename[128];

    snprintf(
        filename,
        sizeof(filename),
        "task1_blocking_node_%d.log",
        world_rank
    );

    FILE *log =
        fopen(
            filename,
            "w"
        );

    if (!log) {

        fprintf(
            stderr,
            "Node %d: log open failed\n",
            world_rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            3
        );
    }

    fprintf(
        log,
        "IMPLEMENTATION: BLOCKING MPI BASELINE\n"
        "World rank: %d\n"
        "Cartesian rank: %d\n"
        "Coordinates: (%d,%d)\n"
        "Neighbour count: %d\n",
        world_rank,
        cart_rank,
        coords[0],
        coords[1],
        neighbour_count
    );

    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        if (world_neighbours[i] !=
            MPI_PROC_NULL) {

            fprintf(
                log,
                "Neighbour direction %d = world rank %d\n",
                i,
                world_neighbours[i]
            );
        }
    }

    fflush(log);

    int completed_cycles = 0;
    int current_available =
        cfg->ports;

    int total_alerts = 0;
    int local_redirections = 0;

    /*
     * In this blocking baseline every valid neighbour
     * exchanges one availability integer in each direction
     * per cycle.
     */
    long long total_neighbour_messages = 0;
    long long total_neighbour_bytes = 0;

    long long exchange_count = 0;

    double total_exchange_time = 0.0;
    double min_exchange_time = DBL_MAX;
    double max_exchange_time = 0.0;

    /*
     * Synchronise the EV-node communicator before
     * beginning the cycle loop.
     *
     * Rank 0 is not part of worker_comm.
     */
    MPI_Barrier(
        worker_comm
    );

    double simulation_start =
        MPI_Wtime();

    for (int cycle = 0;
         cycle < cfg->cycles;
         cycle++) {

        /*
         * Keep every EV node in the same blocking cycle.
         *
         * This synchronisation is deliberately part of
         * the baseline design and contributes to its
         * communication/synchronisation overhead.
         */
        MPI_Barrier(
            worker_comm
        );

        update_ports(
            port_status,
            seeds,
            cfg->ports,
            cfg->force_low,
            cycle
        );

        current_available =
            count_available(
                port_status,
                cfg->ports
            );

        long long timestamp =
            now_epoch_ms();

        char time_string[80];

        format_time(
            timestamp,
            time_string,
            sizeof(time_string)
        );

        fprintf(
            log,
            "\n[%s] CYCLE %d\n"
            "Available: %d/%d\n",
            time_string,
            cycle,
            current_available,
            cfg->ports
        );

        /*
         * All EV nodes participate in a blocking
         * availability exchange so that no process can
         * deadlock waiting for a neighbour that did not
         * enter the protocol.
         */
        int neighbour_available[MAX_NEIGHBOURS];

        double exchange_seconds =
            blocking_neighbour_exchange(
                cart_comm,
                current_available,
                cart_neighbours,
                neighbour_available
            );

        exchange_count++;
        total_exchange_time +=
            exchange_seconds;

        if (exchange_seconds <
            min_exchange_time) {

            min_exchange_time =
                exchange_seconds;
        }

        if (exchange_seconds >
            max_exchange_time) {

            max_exchange_time =
                exchange_seconds;
        }

        /*
         * For every valid edge this node sends one integer
         * and receives one integer.
         */
        total_neighbour_messages +=
            2LL * neighbour_count;

        total_neighbour_bytes +=
            2LL * neighbour_count *
            (long long)sizeof(int);

        fprintf(
            log,
            "Blocking neighbour exchange: %.9f s\n",
            exchange_seconds
        );

        /*
         * Task-1 decision logic:
         *
         * Only a congested node uses neighbour data to
         * choose a local redirect or report congestion.
         */
        if (current_available <=
            cfg->threshold) {

            int best_direction = -1;
            int best_node = -1;
            int best_available = -1;

            for (int i = 0;
                 i < MAX_NEIGHBOURS;
                 i++) {

                if (world_neighbours[i] ==
                    MPI_PROC_NULL) {

                    continue;
                }

                int available =
                    neighbour_available[i];

                if (available >
                    cfg->threshold &&
                    available >
                    best_available) {

                    best_available =
                        available;

                    best_node =
                        world_neighbours[i];

                    best_direction = i;
                }
            }

            if (best_node >= 0) {

                local_redirections++;

                fprintf(
                    log,
                    "LOCAL REDIRECTION\n"
                    "Selected node: %d\n"
                    "Direction: %d\n"
                    "Available ports: %d\n"
                    "Exchange time: %.9f s\n",
                    best_node,
                    best_direction,
                    best_available,
                    exchange_seconds
                );

            } else {

                /*
                 * No immediate neighbour has enough
                 * availability. Report to rank 0.
                 */
                CongestionReport report;

                memset(
                    &report,
                    0,
                    sizeof(report)
                );

                report.node_rank =
                    world_rank;

                report.coord_row =
                    coords[0];

                report.coord_col =
                    coords[1];

                report.total_ports =
                    cfg->ports;

                report.local_available =
                    current_available;

                report.threshold =
                    cfg->threshold;

                /*
                 * Timestamp is taken after neighbour
                 * responses confirm the congestion event.
                 */
                report.epoch_ms =
                    now_epoch_ms();

                report.valid_neighbour_count =
                    neighbour_count;

                report.neighbour_messages =
                    2 * neighbour_count;

                report.neighbour_bytes =
                    2 * neighbour_count *
                    (int)sizeof(int);

                report.neighbour_exchange_seconds =
                    exchange_seconds;

                for (int i = 0;
                     i < MAX_NEIGHBOURS;
                     i++) {

                    report.neighbour_ranks[i] =
                        world_neighbours[i];

                    report.neighbour_available[i] =
                        (world_neighbours[i] ==
                         MPI_PROC_NULL)
                        ? -1
                        : neighbour_available[i];
                }

                /*
                 * Blocking report + blocking reply.
                 */
                MPI_Send(
                    &report,
                    (int)sizeof(report),
                    MPI_BYTE,
                    0,
                    TAG_CONGESTION_REPORT,
                    MPI_COMM_WORLD
                );

                RedirectReply reply;

                memset(
                    &reply,
                    0,
                    sizeof(reply)
                );

                MPI_Recv(
                    &reply,
                    (int)sizeof(reply),
                    MPI_BYTE,
                    0,
                    TAG_REDIRECT,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                total_alerts++;

                fprintf(
                    log,
                    "CONGESTION ALERT SENT\n"
                    "Free ports: %d/%d\n"
                    "Neighbour messages: %d\n"
                    "Neighbour bytes: %d\n"
                    "Exchange time: %.9f s\n"
                    "BASE STATION RESPONSE: %d candidates\n",
                    report.local_available,
                    report.total_ports,
                    report.neighbour_messages,
                    report.neighbour_bytes,
                    exchange_seconds,
                    reply.candidate_count
                );

                for (int i = 0;
                     i < reply.candidate_count &&
                     i < MAX_NEIGHBOURS;
                     i++) {

                    fprintf(
                        log,
                        "Candidate: %d\n",
                        reply.candidate_ranks[i]
                    );
                }
            }
        }

        fflush(log);

        /*
         * Keep cycle timing comparable to the original
         * Task-1 implementation.
         */
        if (cycle + 1 <
            cfg->cycles) {

            sleep_ms(
                cfg->interval_ms
            );
        }

        completed_cycles++;
    }

    /*
     * All scheduled work is complete.
     */
    int done = 1;

    MPI_Send(
        &done,
        1,
        MPI_INT,
        0,
        TAG_DONE,
        MPI_COMM_WORLD
    );

    fprintf(
        log,
        "All local cycles completed.\n"
        "Waiting for termination.\n"
    );

    fflush(log);

    /*
     * The baseline performs a blocking receive for the
     * graceful shutdown command.
     */
    int terminate = 0;

    MPI_Recv(
        &terminate,
        1,
        MPI_INT,
        0,
        TAG_TERMINATE,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    double simulation_time =
        MPI_Wtime() -
        simulation_start;

    double average_exchange =
        exchange_count > 0
        ? total_exchange_time /
          (double)exchange_count
        : 0.0;

    if (exchange_count == 0)
        min_exchange_time = 0.0;

    fprintf(
        log,
        "\n========================================\n"
        "NODE SUMMARY\n"
        "========================================\n"
        "Implementation: BLOCKING MPI\n"
        "Cycles: %d\n"
        "Congestion reports sent: %d\n"
        "Local redirections: %d\n"
        "Neighbour exchanges: %lld\n"
        "Neighbour messages: %lld\n"
        "Neighbour payload bytes: %lld\n"
        "Total neighbour exchange time: %.9f s\n"
        "Average neighbour exchange time: %.9f s\n"
        "Minimum neighbour exchange time: %.9f s\n"
        "Maximum neighbour exchange time: %.9f s\n"
        "Node simulation time: %.9f s\n"
        "Graceful shutdown completed.\n",
        completed_cycles,
        total_alerts,
        local_redirections,
        exchange_count,
        total_neighbour_messages,
        total_neighbour_bytes,
        total_exchange_time,
        average_exchange,
        min_exchange_time,
        max_exchange_time,
        simulation_time
    );

    fclose(log);

    free(
        port_status
    );

    free(
        seeds
    );

    printf(
        "[NODE %d BLOCKING] "
        "Cycles=%d Alerts=%d "
        "AvgExchange=%.6f s\n",
        world_rank,
        completed_cycles,
        total_alerts,
        average_exchange
    );

    fflush(stdout);
}

/* =========================================================
 * MAIN
 * ========================================================= */

int main(
    int argc,
    char **argv
) {
    int provided = 0;

    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int rank;
    int size;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &size
    );

    if (provided <
        MPI_THREAD_FUNNELED) {

        if (rank == 0) {

            fprintf(
                stderr,
                "MPI_THREAD_FUNNELED unavailable\n"
            );
        }

        MPI_Abort(
            MPI_COMM_WORLD,
            4
        );
    }

    if (argc < 3 ||
        argc > 8) {

        if (rank == 0) {

            fprintf(
                stderr,
                "Usage: %s rows cols "
                "[ports] [cycles] [interval_ms] "
                "[threshold] [force_low]\n",
                argv[0]
            );
        }

        MPI_Finalize();

        return 1;
    }

    Config cfg;

    cfg.rows =
        atoi(argv[1]);

    cfg.cols =
        atoi(argv[2]);

    cfg.ports =
        argc > 3
        ? atoi(argv[3])
        : DEFAULT_PORTS;

    cfg.cycles =
        argc > 4
        ? atoi(argv[4])
        : DEFAULT_CYCLES;

    cfg.interval_ms =
        argc > 5
        ? atoi(argv[5])
        : DEFAULT_INTERVAL_MS;

    cfg.threshold =
        argc > 6
        ? atoi(argv[6])
        : DEFAULT_THRESHOLD;

    cfg.force_low =
        argc > 7
        ? (atoi(argv[7]) != 0)
        : 0;

    long long nodes =
        (long long)cfg.rows *
        cfg.cols;

    if (cfg.rows <= 0 ||
        cfg.cols <= 0 ||
        cfg.ports <= 0 ||
        cfg.cycles <= 0 ||
        cfg.interval_ms <= 0 ||
        cfg.threshold < 0 ||
        cfg.threshold >= cfg.ports ||
        nodes > INT_MAX - 1 ||
        nodes + 1 != size) {

        if (rank == 0) {

            fprintf(
                stderr,
                "Invalid configuration.\n"
                "Expected MPI processes = "
                "rows * cols + 1\n"
            );
        }

        MPI_Finalize();

        return 1;
    }

    /*
     * Rank 0 is excluded from the EV-node communicator.
     */
    MPI_Comm worker_comm =
        MPI_COMM_NULL;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        rank == 0
        ? MPI_UNDEFINED
        : 1,
        rank,
        &worker_comm
    );

    /*
     * Synchronise all world processes before starting.
     * This preserves the same startup structure as the
     * reference Task-1 code.
     */
    MPI_Barrier(
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        run_base_station(
            (int)nodes
        );

    } else {

        MPI_Comm cart_comm =
            MPI_COMM_NULL;

        int dims[2] = {
            cfg.rows,
            cfg.cols
        };

        int periods[2] = {
            0,
            0
        };

        /*
         * reorder = 0 so:
         * cart rank = world rank - 1.
         */
        MPI_Cart_create(
            worker_comm,
            2,
            dims,
            periods,
            0,
            &cart_comm
        );

        if (cart_comm ==
            MPI_COMM_NULL) {

            fprintf(
                stderr,
                "Node %d: Cartesian creation failed\n",
                rank
            );

            MPI_Abort(
                MPI_COMM_WORLD,
                5
            );
        }

        run_charging_node(
            rank,
            worker_comm,
            cart_comm,
            &cfg
        );

        MPI_Comm_free(
            &cart_comm
        );

        MPI_Comm_free(
            &worker_comm
        );
    }

    MPI_Finalize();

    return 0;
}
