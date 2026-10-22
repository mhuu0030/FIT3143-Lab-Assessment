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
#define DEFAULT_SEED 3143U
#define DEFAULT_CACHE_MS 2000

/*
 * FIT3143 Lab 3 - Task 1
 * OPTIMISED NON-BLOCKING MPI VERSION
 *
 * Based on lab3_task1.c.
 *
 * Rank 0:
 *     Task-1 base-station test harness.
 *
 * Ranks 1..N:
 *     EV charging nodes arranged in a non-periodic 2-D
 *     MPI Cartesian topology.
 *
 * OpenMP:
 *     Simulates charging ports.
 *
 * MPI:
 *     Only the main thread makes MPI calls
 *     (MPI_THREAD_FUNNELED).
 *
 * ---------------------------------------------------------
 * OPTIMISATIONS COMPARED WITH THE ORIGINAL NON-BLOCKING CODE
 * ---------------------------------------------------------
 *
 * 1. NEIGHBOUR-STATUS CACHE
 *
 *    Each node stores the last availability received from each
 *    neighbour together with the time it was received.
 *
 *    A cached result remains usable for CACHE_TTL milliseconds.
 *
 * 2. QUERY ONLY STALE / UNKNOWN NEIGHBOURS
 *
 *    When the local node is congested, it does NOT automatically
 *    contact every neighbour.
 *
 *    If a neighbour's cached state is still fresh, no request is
 *    sent to that neighbour.
 *
 * 3. ZERO-MESSAGE FAST PATH
 *
 *    If a fresh cached neighbour already has sufficient free
 *    capacity, the node can redirect locally without performing
 *    another MPI neighbour query.
 *
 *    If every neighbour has fresh cached data and all are still
 *    congested, the node can report congestion to the base station
 *    without repeating neighbour communication.
 *
 * 4. NON-BLOCKING REQUEST / RESPONSE TRAFFIC
 *
 *    Query receives are posted with MPI_Irecv before requests are
 *    sent with MPI_Isend.
 *
 *    Incoming neighbour responses are sent with MPI_Isend instead
 *    of blocking the event loop.
 *
 * 5. NON-BLOCKING BASE-STATION REPORT SEND
 *
 *    The redirect receive is posted first, then the congestion
 *    report is submitted with MPI_Isend.
 *
 * 6. EXPLICIT PERFORMANCE METRICS
 *
 *    The log records:
 *      - actual neighbour requests sent
 *      - actual neighbour responses sent
 *      - neighbour requests skipped because cache was fresh
 *      - cache-only redirects
 *      - cache-only congestion decisions
 *      - actual payload bytes
 *      - total / average / min / max query time
 *
 * The cache TTL is configurable as the last command-line argument.
 * A value <= 0 disables caching and makes this version behave much
 * more like the original non-blocking implementation.
 */

enum {
    TAG_NEIGHBOUR_REQUEST  = 100,
    TAG_NEIGHBOUR_RESPONSE = 101,
    TAG_CONGESTION_REPORT  = 200,
    TAG_REDIRECT            = 201,
    TAG_DONE                = 998,
    TAG_TERMINATE           = 999
};

typedef struct {
    int node_rank;
    int coord_row;
    int coord_col;

    int total_ports;
    int local_available;
    int threshold;
    int cycle;

    int valid_neighbour_count;
    int neighbour_ranks[MAX_NEIGHBOURS];
    int neighbour_available[MAX_NEIGHBOURS];
    int neighbour_cycle[MAX_NEIGHBOURS];

    /*
     * For this optimised version these values represent
     * ACTUAL neighbour traffic used for the current incident.
     *
     * If the decision is made entirely from fresh cached
     * information, these can legitimately be zero.
     */
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

    /*
     * How long a neighbour response remains fresh.
     * 0 disables cache reuse.
     */
    int cache_ms;
    unsigned int seed;
} Config;

typedef struct {
    int valid;
    int available;
    int cycle;
    int decision_cycle;
    double received_at;
} NeighbourCache;

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
             * Deterministic congestion for cycle 0
             * so the communication path is exercised.
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
 * CARTESIAN TOPOLOGY
 * ========================================================= */

static int cart_to_world(
    int cart_rank
) {
    if (cart_rank ==
        MPI_PROC_NULL) {

        return MPI_PROC_NULL;
    }

    /*
     * reorder=0 and world rank 0 is excluded,
     * therefore:
     *
     * world rank = Cartesian rank + 1
     */
    return cart_rank + 1;
}

static int discover_neighbours(
    MPI_Comm cart_comm,
    int neighbours[MAX_NEIGHBOURS]
) {
    int top;
    int bottom;
    int left;
    int right;

    MPI_Cart_shift(
        cart_comm,
        0,
        1,
        &top,
        &bottom
    );

    MPI_Cart_shift(
        cart_comm,
        1,
        1,
        &left,
        &right
    );

    int cart_neighbours[MAX_NEIGHBOURS] = {
        top,
        bottom,
        left,
        right
    };

    int count = 0;

    /*
     * Keep the same compact-neighbour representation
     * as the reference implementation.
     */
    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        if (cart_neighbours[i] !=
            MPI_PROC_NULL) {

            neighbours[count++] =
                cart_to_world(
                    cart_neighbours[i]
                );
        }
    }

    for (int i = count;
         i < MAX_NEIGHBOURS;
         i++) {

        neighbours[i] =
            MPI_PROC_NULL;
    }

    return count;
}

/* =========================================================
 * CACHE HELPERS
 * ========================================================= */

static int cache_is_fresh(
    const NeighbourCache *entry,
    double now,
    int cache_ms,
    int required_cycle
) {
    if (!entry->valid)
        return 0;

    if (cache_ms <= 0)
        return 0;

    if (entry->decision_cycle != required_cycle)
        return 0;

    double age =
        now - entry->received_at;

    if (age < 0.0)
        return 0;

    return age <=
        (double)cache_ms / 1000.0;
}

/*
 * Find a fresh cached neighbour that is currently considered
 * available. If more than one exists, choose the one with the
 * greatest free capacity.
 */
static int best_fresh_cached_neighbour(
    const int neighbours[MAX_NEIGHBOURS],
    int neighbour_count,
    const NeighbourCache cache[MAX_NEIGHBOURS],
    double now,
    int cache_ms,
    int required_cycle,
    int threshold,
    int *best_available
) {
    int best_node = -1;
    int most_free = -1;

    for (int i = 0;
         i < neighbour_count;
         i++) {

        if (!cache_is_fresh(
                &cache[i],
                now,
                cache_ms,
                required_cycle)) {

            continue;
        }

        if (cache[i].available >
                threshold &&
            cache[i].available >
                most_free) {

            most_free =
                cache[i].available;

            best_node =
                neighbours[i];
        }
    }

    if (best_available)
        *best_available = most_free;

    return best_node;
}

/*
 * Are ALL valid neighbour cache entries fresh?
 */
static int all_neighbour_cache_fresh(
    int neighbour_count,
    const NeighbourCache cache[MAX_NEIGHBOURS],
    double now,
    int cache_ms,
    int required_cycle
) {
    if (neighbour_count <= 0)
        return 1;

    for (int i = 0;
         i < neighbour_count;
         i++) {

        if (!cache_is_fresh(
                &cache[i],
                now,
                cache_ms,
                required_cycle)) {

            return 0;
        }
    }

    return 1;
}

/* =========================================================
 * TASK 1 BASE-STATION TEST HARNESS
 * ========================================================= */

static void run_base_station(
    int workers
) {
    FILE *log =
        fopen(
            "task1_optimised_base_station.log",
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
        "TASK 1 OPTIMISED NON-BLOCKING TEST HARNESS\n"
        "EV nodes: %d\n\n",
        workers
    );

    fflush(log);

    while (done_count <
           workers) {

        int flag = 0;
        MPI_Status status;

        MPI_Iprobe(
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &flag,
            &status
        );

        if (!flag) {

            /*
             * Avoid burning a whole CPU core in an
             * empty polling loop.
             */
            sleep_ms(1);
            continue;
        }

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
                "Decision cycle: %d\n"
                "Coordinates: (%d,%d)\n"
                "Available ports: %d/%d\n"
                "Threshold: %d\n"
                "Actual neighbour messages this incident: %d\n"
                "Actual neighbour payload bytes this incident: %d\n"
                "Neighbour exchange time: %.9f s\n",
                timestamp,
                report.node_rank,
                report.cycle,
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
                 i <
                 report.valid_neighbour_count;
                 i++) {

                fprintf(
                    log,
                    "Neighbour %d: %d available (cycle %d)\n",
                    report.neighbour_ranks[i],
                    report.neighbour_available[i],
                    report.neighbour_cycle[i]
                );
            }

            /*
             * Task 1 placeholder only.
             * Task 2 replaces this with the actual
             * nearest-available-node search.
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
     * Graceful shutdown.
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
        "Implementation: OPTIMISED NON-BLOCKING MPI\n"
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
        "[BASE OPT] "
        "Completed. Alerts=%d Time=%.3f s\n",
        alerts,
        total
    );

    fflush(stdout);
}

/* =========================================================
 * CONGESTION REPORT HELPER
 * ========================================================= */

static void prepare_congestion_report(
    CongestionReport *report,
    int world_rank,
    const int coords[2],
    const Config *cfg,
    int cycle,
    int local_available,
    int neighbour_count,
    const int neighbours[MAX_NEIGHBOURS],
    const NeighbourCache cache[MAX_NEIGHBOURS],
    int incident_neighbour_messages,
    int incident_neighbour_bytes,
    double exchange_seconds
) {
    memset(
        report,
        0,
        sizeof(*report)
    );

    report->node_rank =
        world_rank;

    report->coord_row =
        coords[0];

    report->coord_col =
        coords[1];

    report->total_ports =
        cfg->ports;

    report->local_available =
        local_available;

    report->threshold =
        cfg->threshold;

    report->cycle = cycle;

    report->valid_neighbour_count =
        neighbour_count;

    /*
     * Timestamp is taken after neighbour evidence
     * has confirmed the congestion state.
     */
    report->epoch_ms =
        now_epoch_ms();

    report->neighbour_messages =
        incident_neighbour_messages;

    report->neighbour_bytes =
        incident_neighbour_bytes;

    report->neighbour_exchange_seconds =
        exchange_seconds;

    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        if (i < neighbour_count) {

            report->neighbour_ranks[i] =
                neighbours[i];

            report->neighbour_available[i] =
                cache[i].valid
                ? cache[i].available
                : -1;

            report->neighbour_cycle[i] =
                cache[i].valid
                ? cache[i].cycle
                : -1;

        } else {

            report->neighbour_ranks[i] =
                MPI_PROC_NULL;

            report->neighbour_available[i] =
                -1;

            report->neighbour_cycle[i] = -1;
        }
    }
}

/* =========================================================
 * TASK 1 EV CHARGING NODE
 * ========================================================= */

static void run_charging_node(
    int world_rank,
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

    int neighbours[MAX_NEIGHBOURS];

    int neighbour_count =
        discover_neighbours(
            cart_comm,
            neighbours
        );

    /*
     * Shared charging-port arrays.
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
            cfg->seed
            ^ ((unsigned int)world_rank *
               1009U)
            ^ ((unsigned int)(i + 1) *
               7919U);
    }

    char filename[128];

    snprintf(
        filename,
        sizeof(filename),
        "task1_optimised_node_%d.log",
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
        "IMPLEMENTATION: OPTIMISED NON-BLOCKING MPI\n"
        "Deterministic seed: %u\n"
        "World rank: %d\n"
        "Cartesian rank: %d\n"
        "Coordinates: (%d,%d)\n"
        "Neighbour count: %d\n"
        "Cache TTL: %d ms\n",
        cfg->seed,
        world_rank,
        cart_rank,
        coords[0],
        coords[1],
        neighbour_count,
        cfg->cache_ms
    );

    for (int i = 0;
         i < neighbour_count;
         i++) {

        fprintf(
            log,
            "Neighbour %d = world rank %d\n",
            i,
            neighbours[i]
        );
    }

    fflush(log);

    /* -----------------------------------------------------
     * Neighbour cache
     * ----------------------------------------------------- */

    NeighbourCache cache[MAX_NEIGHBOURS];

    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        cache[i].valid = 0;
        cache[i].available = -1;
        cache[i].cycle = -1;
        cache[i].decision_cycle = -1;
        cache[i].received_at = 0.0;
    }

    /* -----------------------------------------------------
     * Own enquiry state
     * ----------------------------------------------------- */

    int querying = 0;
    int waiting_redirect = 0;

    /*
     * Query only a subset of neighbours.
     *
     * query_indices[j] maps a query slot back to the
     * corresponding compact neighbour/cache index.
     */
    int query_indices[MAX_NEIGHBOURS];
    int query_count = 0;

    int response_values[MAX_NEIGHBOURS][2];
    int request_payload[MAX_NEIGHBOURS];

    MPI_Request query_sends[MAX_NEIGHBOURS];
    MPI_Request query_receives[MAX_NEIGHBOURS];
    MPI_Status query_send_statuses[MAX_NEIGHBOURS];
    MPI_Status query_receive_statuses[MAX_NEIGHBOURS];

    int enquiry_free_ports = 0;
    int enquiry_cycle = -1;
    double enquiry_start = 0.0;

    /*
     * Actual traffic incurred by the current enquiry.
     */
    int enquiry_messages = 0;
    int enquiry_bytes = 0;

    /* -----------------------------------------------------
     * Base-station report state
     * ----------------------------------------------------- */

    CongestionReport pending_report;
    MPI_Request report_send =
        MPI_REQUEST_NULL;

    RedirectReply redirect_reply;
    MPI_Request redirect_recv =
        MPI_REQUEST_NULL;

    /* -----------------------------------------------------
     * Non-blocking neighbour response sends
     * ----------------------------------------------------- */

    int outgoing_response_value[MAX_NEIGHBOURS][2];

    MPI_Request outgoing_response_send[MAX_NEIGHBOURS];

    for (int i = 0;
         i < MAX_NEIGHBOURS;
         i++) {

        outgoing_response_send[i] =
            MPI_REQUEST_NULL;
    }

    /* -----------------------------------------------------
     * Simulation state
     * ----------------------------------------------------- */

    int completed_cycles = 0;
    int current_available =
        cfg->ports;

    int done_sent = 0;
    int terminated = 0;

    /* -----------------------------------------------------
     * Metrics
     * ----------------------------------------------------- */

    long long total_requests_sent = 0;
    long long total_responses_sent = 0;
    long long total_alerts = 0;
    long long total_local_redirections = 0;

    long long cache_redirects = 0;
    long long cache_only_reports = 0;

    long long requests_skipped_cache = 0;

    long long neighbour_payload_bytes_sent = 0;
    long long neighbour_payload_bytes_received = 0;

    long long completed_queries = 0;

    double total_query_time = 0.0;
    double min_query_time = DBL_MAX;
    double max_query_time = 0.0;

    /*
     * Same startup synchronisation as the reference
     * implementation.
     */
    MPI_Barrier(
        MPI_COMM_WORLD
    );

    double simulation_start =
        MPI_Wtime();

    double next_update =
        MPI_Wtime();

    while (!terminated) {

        /* =================================================
         * A. TERMINATION
         * ================================================= */

        int flag = 0;
        MPI_Status status;

        MPI_Iprobe(
            0,
            TAG_TERMINATE,
            MPI_COMM_WORLD,
            &flag,
            &status
        );

        if (flag) {

            int value = 0;

            MPI_Recv(
                &value,
                1,
                MPI_INT,
                0,
                TAG_TERMINATE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            terminated = 1;
            break;
        }

        /* =================================================
         * B. COMPLETE PREVIOUS NON-BLOCKING RESPONSE SENDS
         * ================================================= */

        for (int i = 0;
             i < neighbour_count;
             i++) {

            if (outgoing_response_send[i] !=
                MPI_REQUEST_NULL) {

                int complete = 0;

                MPI_Test(
                    &outgoing_response_send[i],
                    &complete,
                    MPI_STATUS_IGNORE
                );

                if (complete) {

                    outgoing_response_send[i] =
                        MPI_REQUEST_NULL;
                }
            }
        }

        /* =================================================
         * C. SERVICE INCOMING NEIGHBOUR REQUESTS
         * ================================================= */

        flag = 0;

        MPI_Iprobe(
            MPI_ANY_SOURCE,
            TAG_NEIGHBOUR_REQUEST,
            MPI_COMM_WORLD,
            &flag,
            &status
        );

        if (flag) {

            int request_value = 0;

            MPI_Recv(
                &request_value,
                1,
                MPI_INT,
                status.MPI_SOURCE,
                TAG_NEIGHBOUR_REQUEST,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            /*
             * Find compact neighbour slot so the response
             * gets its own stable send buffer.
             */
            int slot = -1;

            for (int i = 0;
                 i < neighbour_count;
                 i++) {

                if (neighbours[i] ==
                    status.MPI_SOURCE) {

                    slot = i;
                    break;
                }
            }

            if (slot >= 0) {

                /*
                 * If an earlier response to this same
                 * neighbour has not completed yet, finish it
                 * before reusing the buffer.
                 *
                 * This should be rare because each node has
                 * at most one own enquiry outstanding.
                 */
                if (outgoing_response_send[slot] !=
                    MPI_REQUEST_NULL) {

                    MPI_Wait(
                        &outgoing_response_send[slot],
                        MPI_STATUS_IGNORE
                    );

                    outgoing_response_send[slot] =
                        MPI_REQUEST_NULL;
                }

                outgoing_response_value[slot][0] =
                    current_available;

                outgoing_response_value[slot][1] =
                    completed_cycles > 0
                    ? completed_cycles - 1
                    : -1;

                MPI_Isend(
                    &outgoing_response_value[slot],
                    2,
                    MPI_INT,
                    status.MPI_SOURCE,
                    TAG_NEIGHBOUR_RESPONSE,
                    MPI_COMM_WORLD,
                    &outgoing_response_send[slot]
                );

                total_responses_sent++;

                neighbour_payload_bytes_sent +=
                    2 * sizeof(int);

                fprintf(
                    log,
                    "Responded asynchronously to node %d: "
                    "%d free at cycle %d\n",
                    status.MPI_SOURCE,
                    current_available,
                    outgoing_response_value[slot][1]
                );
            }
        }

        /* =================================================
         * D. CHECK OWN NEIGHBOUR QUERY
         * ================================================= */

        if (querying && query_count > 0) {

            int complete = 0;

            MPI_Testall(
                query_count,
                query_receives,
                &complete,
                query_receive_statuses
            );

            if (complete) {

                /*
                 * Ensure request send buffers are reusable.
                 */
                MPI_Waitall(
                    query_count,
                    query_sends,
                    query_send_statuses
                );

                double now =
                    MPI_Wtime();

                double elapsed =
                    now - enquiry_start;

                completed_queries++;

                total_query_time +=
                    elapsed;

                if (elapsed <
                    min_query_time) {

                    min_query_time =
                        elapsed;
                }

                if (elapsed >
                    max_query_time) {

                    max_query_time =
                        elapsed;
                }

                /*
                 * Refresh only the cache entries that were
                 * actually queried.
                 */
                for (int j = 0;
                     j < query_count;
                     j++) {

                    int i =
                        query_indices[j];

                    cache[i].valid = 1;

                    cache[i].available =
                        response_values[j][0];

                    cache[i].cycle =
                        response_values[j][1];

                    cache[i].decision_cycle =
                        enquiry_cycle;

                    cache[i].received_at =
                        now;

                    neighbour_payload_bytes_received +=
                        2 * sizeof(int);
                }

                /*
                 * After refreshing stale neighbours, choose
                 * the best fresh cached neighbour.
                 */
                int best_available = -1;

                int best_node =
                    best_fresh_cached_neighbour(
                        neighbours,
                        neighbour_count,
                        cache,
                        now,
                        cfg->cache_ms,
                        enquiry_cycle,
                        cfg->threshold,
                        &best_available
                    );

                if (best_node >= 0) {

                    total_local_redirections++;

                    fprintf(
                        log,
                        "LOCAL REDIRECTION AFTER PARTIAL QUERY\n"
                        "Selected node: %d\n"
                        "Available ports: %d\n"
                        "Queried neighbours: %d/%d\n"
                        "Exchange time: %.9f s\n",
                        best_node,
                        best_available,
                        query_count,
                        neighbour_count,
                        elapsed
                    );

                    querying = 0;

                } else {

                    /*
                     * Every cache entry should now be fresh:
                     * some were already fresh and the stale
                     * ones have just been queried.
                     */
                    prepare_congestion_report(
                        &pending_report,
                        world_rank,
                        coords,
                        cfg,
                        enquiry_cycle,
                        enquiry_free_ports,
                        neighbour_count,
                        neighbours,
                        cache,
                        enquiry_messages,
                        enquiry_bytes,
                        elapsed
                    );

                    memset(
                        &redirect_reply,
                        0,
                        sizeof(redirect_reply)
                    );

                    /*
                     * Receive first.
                     */
                    MPI_Irecv(
                        &redirect_reply,
                        (int)sizeof(redirect_reply),
                        MPI_BYTE,
                        0,
                        TAG_REDIRECT,
                        MPI_COMM_WORLD,
                        &redirect_recv
                    );

                    /*
                     * Then send report asynchronously.
                     */
                    MPI_Isend(
                        &pending_report,
                        (int)sizeof(pending_report),
                        MPI_BYTE,
                        0,
                        TAG_CONGESTION_REPORT,
                        MPI_COMM_WORLD,
                        &report_send
                    );

                    waiting_redirect = 1;
                    total_alerts++;

                    fprintf(
                        log,
                        "CONGESTION ALERT SENT\n"
                        "Free ports: %d/%d\n"
                        "Actually queried neighbours: %d/%d\n"
                        "Actual neighbour messages: %d\n"
                        "Actual neighbour bytes: %d\n"
                        "Exchange time: %.9f s\n",
                        pending_report.local_available,
                        pending_report.total_ports,
                        query_count,
                        neighbour_count,
                        pending_report.neighbour_messages,
                        pending_report.neighbour_bytes,
                        elapsed
                    );

                    querying = 0;
                }
            }
        }

        /* =================================================
         * E. BASE-STATION RESPONSE
         * ================================================= */

        if (waiting_redirect) {

            int complete = 0;

            MPI_Test(
                &redirect_recv,
                &complete,
                MPI_STATUS_IGNORE
            );

            if (complete) {

                /*
                 * Ensure the non-blocking report send has
                 * completed before pending_report can ever
                 * be reused.
                 */
                if (report_send !=
                    MPI_REQUEST_NULL) {

                    MPI_Wait(
                        &report_send,
                        MPI_STATUS_IGNORE
                    );

                    report_send =
                        MPI_REQUEST_NULL;
                }

                fprintf(
                    log,
                    "BASE STATION RESPONSE: "
                    "%d candidates\n",
                    redirect_reply.candidate_count
                );

                for (int i = 0;
                     i <
                     redirect_reply.candidate_count &&
                     i < MAX_NEIGHBOURS;
                     i++) {

                    fprintf(
                        log,
                        "Candidate: %d\n",
                        redirect_reply.candidate_ranks[i]
                    );
                }

                waiting_redirect = 0;
            }
        }

        /* =================================================
         * F. PERIODIC PORT UPDATE
         * ================================================= */

        double now =
            MPI_Wtime();

        if (completed_cycles <
                cfg->cycles &&
            !querying &&
            !waiting_redirect &&
            now >= next_update) {

            int cycle =
                completed_cycles;

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

            completed_cycles++;

            next_update =
                MPI_Wtime() +
                cfg->interval_ms /
                1000.0;

            /*
             * No communication at all when local
             * availability is above the threshold.
             */
            if (current_available <=
                cfg->threshold) {

                enquiry_free_ports =
                    current_available;

                enquiry_cycle = cycle;

                now =
                    MPI_Wtime();

                /*
                 * -------- FAST PATH 1 --------------------
                 * Reuse fresh cached availability.
                 *
                 * If a known fresh neighbour is already
                 * available, redirect without sending any
                 * new MPI neighbour messages.
                 */
                int cached_best_available = -1;

                int cached_best_node =
                    best_fresh_cached_neighbour(
                        neighbours,
                        neighbour_count,
                        cache,
                        now,
                        cfg->cache_ms,
                        enquiry_cycle,
                        cfg->threshold,
                        &cached_best_available
                    );

                if (cached_best_node >= 0) {

                    total_local_redirections++;
                    cache_redirects++;

                    /*
                     * Every fresh cached entry that avoids a
                     * request counts as one skipped request.
                     * Only count valid neighbours here.
                     */
                    for (int i = 0;
                         i < neighbour_count;
                         i++) {

                        if (cache_is_fresh(
                                &cache[i],
                                now,
                                cfg->cache_ms,
                                enquiry_cycle)) {

                            requests_skipped_cache++;
                        }
                    }

                    fprintf(
                        log,
                        "CACHE-ONLY LOCAL REDIRECTION\n"
                        "Selected node: %d\n"
                        "Available ports: %d\n"
                        "Neighbour messages this decision: 0\n",
                        cached_best_node,
                        cached_best_available
                    );

                } else if (
                    all_neighbour_cache_fresh(
                        neighbour_count,
                        cache,
                        now,
                        cfg->cache_ms,
                        enquiry_cycle)) {

                    /*
                     * -------- FAST PATH 2 ----------------
                     *
                     * All neighbour states are fresh and
                     * none is available. Re-querying them
                     * would duplicate recent communication.
                     */
                    for (int i = 0;
                         i < neighbour_count;
                         i++) {

                        requests_skipped_cache++;
                    }

                    prepare_congestion_report(
                        &pending_report,
                        world_rank,
                        coords,
                        cfg,
                        enquiry_cycle,
                        current_available,
                        neighbour_count,
                        neighbours,
                        cache,
                        0,
                        0,
                        0.0
                    );

                    memset(
                        &redirect_reply,
                        0,
                        sizeof(redirect_reply)
                    );

                    MPI_Irecv(
                        &redirect_reply,
                        (int)sizeof(redirect_reply),
                        MPI_BYTE,
                        0,
                        TAG_REDIRECT,
                        MPI_COMM_WORLD,
                        &redirect_recv
                    );

                    MPI_Isend(
                        &pending_report,
                        (int)sizeof(pending_report),
                        MPI_BYTE,
                        0,
                        TAG_CONGESTION_REPORT,
                        MPI_COMM_WORLD,
                        &report_send
                    );

                    waiting_redirect = 1;

                    total_alerts++;
                    cache_only_reports++;

                    fprintf(
                        log,
                        "CACHE-ONLY CONGESTION ALERT SENT\n"
                        "Free ports: %d/%d\n"
                        "Neighbour messages this incident: 0\n"
                        "Neighbour bytes this incident: 0\n",
                        current_available,
                        cfg->ports
                    );

                } else if (neighbour_count >
                           0) {

                    /*
                     * -------- PARTIAL QUERY ---------------
                     *
                     * Contact ONLY stale or unknown
                     * neighbours.
                     */
                    query_count = 0;

                    enquiry_messages = 0;
                    enquiry_bytes = 0;

                    enquiry_start =
                        MPI_Wtime();

                    for (int i = 0;
                         i < neighbour_count;
                         i++) {

                        if (cache_is_fresh(
                                &cache[i],
                                enquiry_start,
                                cfg->cache_ms,
                                enquiry_cycle)) {

                            /*
                             * Fresh cached congested
                             * neighbour: no new request.
                             */
                            requests_skipped_cache++;
                            continue;
                        }

                        int j =
                            query_count++;

                        query_indices[j] = i;
                        response_values[j][0] = -1;
                        response_values[j][1] = -1;

                        MPI_Irecv(
                            &response_values[j],
                            2,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_RESPONSE,
                            MPI_COMM_WORLD,
                            &query_receives[j]
                        );
                    }

                    /*
                     * Receives are posted before sends.
                     */
                    for (int j = 0;
                         j < query_count;
                         j++) {

                        int i =
                            query_indices[j];

                        request_payload[j] =
                            world_rank;

                        MPI_Isend(
                            &request_payload[j],
                            1,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_REQUEST,
                            MPI_COMM_WORLD,
                            &query_sends[j]
                        );

                        total_requests_sent++;

                        neighbour_payload_bytes_sent +=
                            sizeof(int);
                    }

                    enquiry_messages =
                        2 * query_count;

                    enquiry_bytes =
                        3 * query_count *
                        (int)sizeof(int);

                    querying = 1;

                    fprintf(
                        log,
                        "PARTIAL NON-BLOCKING QUERY STARTED\n"
                        "Queried neighbours: %d/%d\n"
                        "Requests avoided via cache: %d\n",
                        query_count,
                        neighbour_count,
                        neighbour_count -
                            query_count
                    );

                } else {

                    /*
                     * 1x1 grid:
                     * no neighbour traffic is possible.
                     */
                    prepare_congestion_report(
                        &pending_report,
                        world_rank,
                        coords,
                        cfg,
                        cycle,
                        current_available,
                        0,
                        neighbours,
                        cache,
                        0,
                        0,
                        0.0
                    );

                    memset(
                        &redirect_reply,
                        0,
                        sizeof(redirect_reply)
                    );

                    MPI_Irecv(
                        &redirect_reply,
                        (int)sizeof(redirect_reply),
                        MPI_BYTE,
                        0,
                        TAG_REDIRECT,
                        MPI_COMM_WORLD,
                        &redirect_recv
                    );

                    MPI_Isend(
                        &pending_report,
                        (int)sizeof(pending_report),
                        MPI_BYTE,
                        0,
                        TAG_CONGESTION_REPORT,
                        MPI_COMM_WORLD,
                        &report_send
                    );

                    waiting_redirect = 1;
                    total_alerts++;
                }
            }
        }

        /* =================================================
         * G. DONE / SHUTDOWN HANDSHAKE
         * ================================================= */

        if (completed_cycles >=
                cfg->cycles &&
            !querying &&
            !waiting_redirect &&
            !done_sent) {

            int done = 1;

            MPI_Send(
                &done,
                1,
                MPI_INT,
                0,
                TAG_DONE,
                MPI_COMM_WORLD
            );

            done_sent = 1;

            fprintf(
                log,
                "All local cycles completed.\n"
                "Waiting for termination.\n"
            );
        }

        fflush(log);

        /*
         * Small sleep prevents the event loop from
         * busy-spinning when there is no MPI work.
         */
        sleep_ms(1);
    }

    /*
     * Finish any outstanding asynchronous neighbour
     * response sends before releasing their buffers.
     */
    for (int i = 0;
         i < neighbour_count;
         i++) {

        if (outgoing_response_send[i] !=
            MPI_REQUEST_NULL) {

            MPI_Wait(
                &outgoing_response_send[i],
                MPI_STATUS_IGNORE
            );
        }
    }

    if (report_send !=
        MPI_REQUEST_NULL) {

        MPI_Wait(
            &report_send,
            MPI_STATUS_IGNORE
        );
    }

    double simulation_time =
        MPI_Wtime() -
        simulation_start;

    double average_query_time =
        completed_queries > 0
        ? total_query_time /
          (double)completed_queries
        : 0.0;

    if (completed_queries == 0)
        min_query_time = 0.0;

    long long total_neighbour_messages =
        total_requests_sent +
        total_responses_sent;

    long long total_neighbour_bytes =
        neighbour_payload_bytes_sent +
        neighbour_payload_bytes_received;

    fprintf(
        log,
        "\n========================================\n"
        "NODE SUMMARY\n"
        "========================================\n"
        "Implementation: OPTIMISED NON-BLOCKING MPI\n"
        "Cycles: %d\n"
        "Cache TTL: %d ms\n"
        "Neighbour requests sent: %lld\n"
        "Neighbour responses sent: %lld\n"
        "Total actual neighbour messages: %lld\n"
        "Neighbour payload bytes sent: %lld\n"
        "Neighbour payload bytes received: %lld\n"
        "Total neighbour payload bytes: %lld\n"
        "Requests skipped because cache was fresh: %lld\n"
        "Cache-only local redirections: %lld\n"
        "Cache-only congestion reports: %lld\n"
        "All local redirections: %lld\n"
        "Congestion reports sent: %lld\n"
        "Completed MPI neighbour queries: %lld\n"
        "Total neighbour-query time: %.9f s\n"
        "Average neighbour-query time: %.9f s\n"
        "Minimum neighbour-query time: %.9f s\n"
        "Maximum neighbour-query time: %.9f s\n"
        "Node simulation time: %.9f s\n"
        "Graceful shutdown completed.\n",
        completed_cycles,
        cfg->cache_ms,
        total_requests_sent,
        total_responses_sent,
        total_neighbour_messages,
        neighbour_payload_bytes_sent,
        neighbour_payload_bytes_received,
        total_neighbour_bytes,
        requests_skipped_cache,
        cache_redirects,
        cache_only_reports,
        total_local_redirections,
        total_alerts,
        completed_queries,
        total_query_time,
        average_query_time,
        min_query_time,
        max_query_time,
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
        "[NODE %d OPT] "
        "Cycles=%d Alerts=%lld "
        "Requests=%lld Skipped=%lld\n",
        world_rank,
        completed_cycles,
        total_alerts,
        total_requests_sent,
        requests_skipped_cache
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
        argc > 10) {

        if (rank == 0) {

            fprintf(
                stderr,
                "Usage: %s rows cols "
                "[ports] [cycles] [interval_ms] "
                "[threshold] [force_low] [cache_ms] [seed]\n",
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

    cfg.cache_ms =
        argc > 8
        ? atoi(argv[8])
        : DEFAULT_CACHE_MS;

    cfg.seed =
        argc > 9
        ? (unsigned int)strtoul(argv[9], NULL, 10)
        : DEFAULT_SEED;

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
        cfg.cache_ms < 0 ||
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
     * Rank 0 is the base station.
     * Ranks 1..N are EV nodes.
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

    if (rank == 0) {

        MPI_Barrier(
            MPI_COMM_WORLD
        );

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
         * reorder=0 preserves:
         *
         * cart rank = world rank - 1
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
