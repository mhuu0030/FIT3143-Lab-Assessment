
#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <float.h>

#define MAX_NEIGHBOURS 4

#define DEFAULT_PORTS 6
#define DEFAULT_CYCLES 8
#define DEFAULT_INTERVAL_MS 1000
#define DEFAULT_THRESHOLD 1
#define DEFAULT_REPORT_PERIOD_MS 2000
#define DEFAULT_SEED 3143U

/*
 * FIT3143 Lab 3 - Tasks 1 and 2
 *
 * World rank 0: Base station
 * World ranks 1..N: EV charging nodes
 *
 * EV nodes form a nonperiodic Cartesian grid.
 * Cartesian rank = World rank - 1.
 *
 * OpenMP simulates charging ports.
 * MPI is called only by the main thread.
 */

enum {
    TAG_NEIGHBOUR_REQUEST  = 100,
    TAG_NEIGHBOUR_RESPONSE = 101,
    TAG_NODE_STATUS_UPDATE = 150,
    TAG_CONGESTION_REPORT  = 200,
    TAG_REDIRECT_COUNT     = 201,
    TAG_REDIRECT_LIST      = 202,
    TAG_DONE               = 998,
    TAG_TERMINATE          = 999
};

typedef struct {
    int rows;
    int cols;
    int ports;
    int cycles;
    int interval_ms;
    int threshold;
    int force_low;
    int report_period_ms;
    unsigned int seed;
} Config;

/*
 * A node's current status:
 * rank, number of free ports, cycle number.
 */
typedef struct {
    int rank;
    int free_ports;
    int cycle;
} StatusUpdate;

/*
 * Complete congestion report.
 */
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

/*
 * Base station's latest direct node status.
 */
typedef struct {
    int known;
    int free_ports;
    int cycle;
    double received_at;
} NodeStatus;

/*
 * Base station's latest congestion report.
 *
 * This history is separate from node status.
 * A later recovery status does not remove
 * an earlier congestion report.
 */
typedef struct {
    int known;
    int cycle;
    double received_at;
    CongestionReport report;
} NodeHistory;

typedef struct {
    long long status_updates;
    long long alerts;
    long long invalid_reports;
    long long done_messages;
    long long redirect_counts;
    long long redirect_lists;
    long long termination_messages;

    long long received_bytes;
    long long sent_bytes;

    long long with_candidates;
    long long without_candidates;

    double total_processing_time;
} BaseMetrics;

/* =========================================================
 * COMMON UTILITIES
 * ========================================================= */

static void sleep_ms(int ms)
{
    if (ms <= 0)
        return;

    struct timespec req;

    req.tv_sec = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;

    while (nanosleep(&req, &req) == -1 &&
           errno == EINTR) {
        /* Continue sleeping if interrupted. */
    }
}

static long long epoch_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    return (long long)ts.tv_sec * 1000LL
         + ts.tv_nsec / 1000000LL;
}

static void format_time(
    long long timestamp,
    char *buffer,
    size_t capacity
) {
    time_t seconds = (time_t)(timestamp / 1000LL);
    struct tm t;
    char temp[64];

    localtime_r(&seconds, &t);

    strftime(
        temp, sizeof(temp),
        "%Y-%m-%d %H:%M:%S", &t
    );

    snprintf(
        buffer, capacity, "%s.%03lld",
        temp, timestamp % 1000LL
    );
}

static int node_row(int rank, int cols)
{
    return (rank - 1) / cols;
}

static int node_col(int rank, int cols)
{
    return (rank - 1) % cols;
}

static int manhattan_distance(
    int a,
    int b,
    int cols
) {
    return abs(node_row(a, cols) -
               node_row(b, cols))
         + abs(node_col(a, cols) -
               node_col(b, cols));
}

/*
 * Immediate neighbours in the order:
 * top, bottom, left, right.
 */
static int grid_neighbours(
    int rank,
    const Config *cfg,
    int result[MAX_NEIGHBOURS]
) {
    int row = node_row(rank, cfg->cols);
    int col = node_col(rank, cfg->cols);
    int n = 0;

    if (row > 0)
        result[n++] = rank - cfg->cols;

    if (row + 1 < cfg->rows)
        result[n++] = rank + cfg->cols;

    if (col > 0)
        result[n++] = rank - 1;

    if (col + 1 < cfg->cols)
        result[n++] = rank + 1;

    return n;
}

/* =========================================================
 * TASK 1: OPENMP PORT SIMULATION
 * ========================================================= */

/*
 * 1 = free
 * 0 = occupied
 *
 * Every port has a separate random seed.
 */
static void update_ports(
    int *ports,
    unsigned int *seeds,
    int count,
    int force_low,
    int cycle
) {
    int free_probability = force_low ? 10 : 55;

    /*
     * Exactly one OpenMP thread is requested per charging port.
     * Thread i owns port i for this update.
     */
    #pragma omp parallel num_threads(count) default(none) \
        shared(ports, seeds, count, force_low, cycle, free_probability)
    {
        int i = omp_get_thread_num();

        if (i < count) {
            if (force_low && cycle == 0) {
                ports[i] = 0;
            } else {
                unsigned int random_value =
                    rand_r(&seeds[i]) % 100U;

                ports[i] =
                    random_value <
                    (unsigned int)free_probability;
            }
        }
    }
}

static int count_free_ports(
    const int *ports,
    int count
) {
    int free_count = 0;

    for (int i = 0; i < count; i++) {
        free_count += ports[i];
    }

    return free_count;
}

static int cart_neighbours(
    MPI_Comm cart_comm,
    int result[MAX_NEIGHBOURS]
) {
    int top, bottom, left, right;

    MPI_Cart_shift(
        cart_comm, 0, 1, &top, &bottom
    );

    MPI_Cart_shift(
        cart_comm, 1, 1, &left, &right
    );

    int ranks[4] = {
        top, bottom, left, right
    };

    int n = 0;

    for (int i = 0; i < 4; i++) {
        if (ranks[i] != MPI_PROC_NULL) {
            /* reorder=0, world rank = cart rank+1 */
            result[n++] = ranks[i] + 1;
        }
    }

    return n;
}

/* =========================================================
 * TASK 2: REPORT HISTORY
 * ========================================================= */

/*
 * Update direct status only if the cycle
 * is newer than the recorded cycle.
 *
 * A stale status update cannot overwrite
 * a newer direct status.
 */
static void save_status(
    NodeStatus *statuses,
    int rank,
    int free_ports,
    int cycle,
    double receive_time
) {
    NodeStatus *s = &statuses[rank];

    if (!s->known || cycle > s->cycle) {
        s->known = 1;
        s->free_ports = free_ports;
        s->cycle = cycle;
        s->received_at = receive_time;
    }
}

/*
 * EXACT TASK 2 REPORT-HISTORY RULE.
 *
 * Did this node submit a congestion report
 * within the predefined reporting period?
 *
 * IMPORTANT:
 * A later recovery status does not erase
 * the fact that a report was submitted.
 */
static int has_report_within_period(
    int node,
    const NodeHistory *history,
    double now,
    double period_seconds
) {
    return history[node].known &&
           now >= history[node].received_at &&
           now - history[node].received_at
               <= period_seconds;
}

/*
 * Candidate eligibility is different from
 * report-reception history.
 *
 * A node with a recent congestion report
 * is excluded unless a newer direct status
 * shows that it has recovered.
 */
static int currently_congested(
    int node,
    const NodeHistory *history,
    const NodeStatus *statuses,
    double now,
    double period,
    int threshold
) {
    if (!has_report_within_period(
            node, history, now, period)) {
        return 0;
    }

    if (statuses[node].known &&
        statuses[node].cycle >
            history[node].cycle &&
        statuses[node].free_ports > threshold) {

        return 0;
    }

    return 1;
}

/*
 * Validate a congestion report.
 */
static int validate_report(
    const CongestionReport *r,
    int sender,
    const Config *cfg
) {
    int nodes = cfg->rows * cfg->cols;

    if (sender < 1 || sender > nodes)
        return 0;

    if (r->node_rank != sender)
        return 0;

    if (r->row != node_row(sender, cfg->cols) ||
        r->col != node_col(sender, cfg->cols))
        return 0;

    if (r->cycle < 0 || r->cycle >= cfg->cycles)
        return 0;

    if (r->total_ports != cfg->ports ||
        r->threshold != cfg->threshold)
        return 0;

    if (r->free_ports < 0 ||
        r->free_ports > cfg->threshold)
        return 0;

    int expected[MAX_NEIGHBOURS];

    int expected_count =
        grid_neighbours(sender, cfg, expected);

    if (r->neighbour_count != expected_count)
        return 0;

    for (int i = 0; i < expected_count; i++) {

        if (r->neighbour_rank[i] != expected[i])
            return 0;

        if (r->neighbour_free[i] < 0 ||
            r->neighbour_free[i] > cfg->threshold)
            return 0;

        if (r->neighbour_cycle[i] < -1 ||
            r->neighbour_cycle[i] >= cfg->cycles)
            return 0;
    }

    if (r->neighbour_messages !=
        2 * expected_count)
        return 0;

    if (r->neighbour_bytes !=
        3 * expected_count * (int)sizeof(int))
        return 0;

    return 1;
}

/* =========================================================
 * TASK 2: NEAREST AVAILABLE NODE SEARCH
 * ========================================================= */

/*
 * Task 2 selection algorithm:
 *
 * STEP 1
 * Check whether EVERY immediate neighbour
 * submitted a congestion report within
 * the predefined period.
 *
 * If yes, send no nearby alternatives, as required by Task 2.
 *
 * STEP 2
 * Otherwise, search beyond immediate
 * neighbours using Manhattan distance.
 *
 * STEP 3
 * An alternative must:
 * - have recent direct status information
 * - have free ports > congestion threshold
 * - not be currently classified congested
 *
 * STEP 4
 * Return ALL alternatives at the minimum
 * eligible distance.
 */
static int find_nearest_candidates(
    int reporting_node,
    const Config *cfg,
    const NodeStatus *statuses,
    const NodeHistory *history,
    double now,
    int *candidates,
    int *nearest_distance,
    int *all_neighbours_reported
) {
    int total_nodes = cfg->rows * cfg->cols;

    double period =
        cfg->report_period_ms / 1000.0;

    int neighbours[MAX_NEIGHBOURS];

    int n = grid_neighbours(
        reporting_node, cfg, neighbours
    );

    int all_reported = n > 0;

    /*
     * CORRECTED RULE:
     * Check receipt of reports, not
     * whether the nodes recovered later.
     */
    for (int i = 0; i < n; i++) {

        if (!has_report_within_period(
                neighbours[i],
                history,
                now,
                period)) {

            all_reported = 0;
            break;
        }
    }

    *all_neighbours_reported = all_reported;
    *nearest_distance = -1;

    if (all_reported) {
        return 0;
    }

    int closest_distance = INT_MAX;
    int result_count = 0;

    for (int node = 1;
         node <= total_nodes;
         node++) {

        if (node == reporting_node)
            continue;

        int distance = manhattan_distance(
            reporting_node, node, cfg->cols
        );

        /*
         * Task 1 already checks
         * immediate neighbours.
         */
        if (distance < 2)
            continue;

        const NodeStatus *s = &statuses[node];

        if (!s->known)
            continue;

        if (now < s->received_at ||
            now - s->received_at > period)
            continue;

        if (s->free_ports <= cfg->threshold)
            continue;

        if (currently_congested(
                node,
                history,
                statuses,
                now,
                period,
                cfg->threshold))
            continue;

        if (distance < closest_distance) {

            closest_distance = distance;
            result_count = 0;

            candidates[result_count++] = node;

        } else if (distance == closest_distance) {

            candidates[result_count++] = node;
        }
    }

    if (result_count > 0)
        *nearest_distance = closest_distance;

    return result_count;
}

/*
 * Explain why a candidate was rejected.
 */
static const char *rejection_reason(
    int node,
    int reporting_node,
    const Config *cfg,
    const NodeStatus *statuses,
    const NodeHistory *history,
    double now,
    int nearest_distance,
    int all_neighbours_reported
) {
    double period =
        cfg->report_period_ms / 1000.0;

    int distance = manhattan_distance(
        reporting_node, node, cfg->cols
    );

    if (all_neighbours_reported)
        return "all immediate neighbours reported";

    if (!statuses[node].known)
        return "unknown availability";

    if (now < statuses[node].received_at ||
        now - statuses[node].received_at > period)
        return "stale availability";

    if (statuses[node].free_ports <= cfg->threshold)
        return "insufficient free ports";

    if (currently_congested(
            node, history, statuses,
            now, period, cfg->threshold))
        return "recent congestion";

    if (nearest_distance >= 0 &&
        distance > nearest_distance)
        return "farther than nearest candidates";

    return "not selected";
}

/* =========================================================
 * TASK 2: BASE STATION
 * ========================================================= */

static void run_base_station(const Config *cfg)
{
    int node_count = cfg->rows * cfg->cols;

    FILE *log = fopen(
        "task2_nonblocking_base_station.log", "w"
    );

    if (!log) {
        perror("Base station log");
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    NodeStatus *statuses = calloc(
        (size_t)node_count + 1,
        sizeof(NodeStatus)
    );

    NodeHistory *history = calloc(
        (size_t)node_count + 1,
        sizeof(NodeHistory)
    );

    unsigned char *finished = calloc(
        (size_t)node_count + 1,
        sizeof(unsigned char)
    );

    int *candidates = malloc(
        (size_t)node_count * sizeof(int)
    );

    if (!statuses || !history ||
        !finished || !candidates) {

        fprintf(stderr,
                "[BASE] Memory allocation failed\n");

        MPI_Abort(MPI_COMM_WORLD, 11);
    }

    BaseMetrics metrics = {0};
    int done_count = 0;

    double simulation_start = MPI_Wtime();

    fprintf(log,
        "========================================\n"
        "FIT3143 LAB 3 - TASK 2 BASE STATION\n"
        "Implementation: NONBLOCKING MPI\n"
        "========================================\n"
        "Grid: %d x %d\n"
        "EV nodes: %d\n"
        "Ports per node: %d\n"
        "Simulation cycles: %d\n"
        "Update interval: %d ms\n"
        "Threshold: %d\n"
        "Report history period: %d ms\n"
        "Deterministic seed: %u\n\n",
        cfg->rows,
        cfg->cols,
        node_count,
        cfg->ports,
        cfg->cycles,
        cfg->interval_ms,
        cfg->threshold,
        cfg->report_period_ms,
        cfg->seed
    );

    /*
     * One main communication thread.
     *
     * Continue processing messages until
     * every charging node sends DONE.
     */
    while (done_count < node_count) {

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
            sleep_ms(1);
            continue;
        }

        int sender = status.MPI_SOURCE;

        /* --------------------------------------
         * A. PERIODIC NODE STATUS UPDATE
         * -------------------------------------- */
        if (status.MPI_TAG ==
            TAG_NODE_STATUS_UPDATE) {

            StatusUpdate update;

            MPI_Recv(
                &update,
                (int)sizeof(update),
                MPI_BYTE,
                sender,
                TAG_NODE_STATUS_UPDATE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            metrics.status_updates++;
            metrics.received_bytes +=
                sizeof(update);

            if (sender >= 1 &&
                sender <= node_count &&
                update.rank == sender &&
                update.free_ports >= 0 &&
                update.free_ports <= cfg->ports &&
                update.cycle >= 0 &&
                update.cycle < cfg->cycles) {

                save_status(
                    statuses,
                    sender,
                    update.free_ports,
                    update.cycle,
                    MPI_Wtime()
                );
            }

        /* --------------------------------------
         * B. CONGESTION REPORT
         * -------------------------------------- */
        } else if (status.MPI_TAG ==
                   TAG_CONGESTION_REPORT) {

            CongestionReport report;
            memset(&report, 0, sizeof(report));

            MPI_Recv(
                &report,
                (int)sizeof(report),
                MPI_BYTE,
                sender,
                TAG_CONGESTION_REPORT,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            double start = MPI_Wtime();

            metrics.alerts++;
            metrics.received_bytes +=
                sizeof(report);

            int valid = validate_report(
                &report, sender, cfg
            );

            int candidate_count = 0;
            int nearest_distance = -1;
            int all_reported = 0;

            if (valid) {

                /*
                 * Save latest report without
                 * replacing a later cycle.
                 */
                if (!history[sender].known ||
                    report.cycle >=
                        history[sender].cycle) {

                    history[sender].known = 1;
                    history[sender].cycle =
                        report.cycle;
                    history[sender].received_at =
                        start;
                    history[sender].report =
                        report;
                }

                /*
                 * A congestion report also
                 * provides direct availability
                 * for the reporting station.
                 */
                save_status(
                    statuses,
                    sender,
                    report.free_ports,
                    report.cycle,
                    start
                );

                candidate_count =
                    find_nearest_candidates(
                        sender,
                        cfg,
                        statuses,
                        history,
                        start,
                        candidates,
                        &nearest_distance,
                        &all_reported
                    );

            } else {
                metrics.invalid_reports++;
            }

            /*
             * Send candidate count first.
             * Then send variable-sized list.
             */
            MPI_Send(
                &candidate_count,
                1,
                MPI_INT,
                sender,
                TAG_REDIRECT_COUNT,
                MPI_COMM_WORLD
            );

            metrics.redirect_counts++;
            metrics.sent_bytes += sizeof(int);

            if (candidate_count > 0) {

                MPI_Send(
                    candidates,
                    candidate_count,
                    MPI_INT,
                    sender,
                    TAG_REDIRECT_LIST,
                    MPI_COMM_WORLD
                );

                metrics.redirect_lists++;

                metrics.sent_bytes +=
                    (long long)candidate_count *
                    sizeof(int);
            }

            double elapsed = MPI_Wtime() - start;
            metrics.total_processing_time += elapsed;

            if (candidate_count > 0)
                metrics.with_candidates++;
            else
                metrics.without_candidates++;

            char timestamp[80];

            format_time(
                report.timestamp_ms,
                timestamp,
                sizeof(timestamp)
            );

            fprintf(log,
                "----------------------------------------\n"
                "INCIDENT %lld\n"
                "Timestamp: %s\n"
                "Sender: %d\n"
                "Report valid: %s\n"
                "Cycle: %d\n"
                "Coordinates: (%d,%d)\n"
                "Local availability: %d/%d\n"
                "Neighbour messages: %d\n"
                "Neighbour payload bytes: %d\n"
                "Neighbour exchange: %.9f s\n",
                metrics.alerts,
                timestamp,
                sender,
                valid ? "YES" : "NO",
                report.cycle,
                report.row,
                report.col,
                report.free_ports,
                report.total_ports,
                report.neighbour_messages,
                report.neighbour_bytes,
                report.neighbour_exchange_seconds
            );

            int n = report.neighbour_count;

            if (n < 0)
                n = 0;

            if (n > MAX_NEIGHBOURS)
                n = MAX_NEIGHBOURS;

            for (int i = 0; i < n; i++) {

                fprintf(log,
                    "Neighbour %d: %d free "
                    "(cycle %d)\n",
                    report.neighbour_rank[i],
                    report.neighbour_free[i],
                    report.neighbour_cycle[i]
                );
            }

            fprintf(log,
                "All immediate neighbours "
                "reported within period: %s\n"
                "Nearest distance: %d\n"
                "Candidate count: %d\n",
                all_reported ? "YES" : "NO",
                nearest_distance,
                candidate_count
            );

            /*
             * Log accepted candidate nodes.
             */
            for (int i = 0;
                 i < candidate_count;
                 i++) {

                int node = candidates[i];

                fprintf(log,
                    "ACCEPT node=%d "
                    "coord=(%d,%d) "
                    "free=%d distance=%d\n",
                    node,
                    node_row(node, cfg->cols),
                    node_col(node, cfg->cols),
                    statuses[node].free_ports,
                    nearest_distance
                );
            }

            /*
             * Log rejected candidate nodes.
             */
            if (valid) {

                double evaluation_time = start;

                for (int node = 1;
                     node <= node_count;
                     node++) {

                    int distance =
                        manhattan_distance(
                            sender, node, cfg->cols
                        );

                    if (distance < 2)
                        continue;

                    int selected = 0;

                    for (int i = 0;
                         i < candidate_count;
                         i++) {

                        if (candidates[i] == node) {
                            selected = 1;
                            break;
                        }
                    }

                    if (selected)
                        continue;

                    fprintf(log,
                        "REJECT node=%d reason=%s\n",
                        node,
                        rejection_reason(
                            node,
                            sender,
                            cfg,
                            statuses,
                            history,
                            evaluation_time,
                            nearest_distance,
                            all_reported
                        )
                    );
                }
            }

            fprintf(log,
                "Processing time: %.9f s\n\n",
                elapsed
            );

            fflush(log);

        /* --------------------------------------
         * C. NODE FINISHED
         * -------------------------------------- */
        } else if (status.MPI_TAG == TAG_DONE) {

            int done_value = 0;

            MPI_Recv(
                &done_value,
                1,
                MPI_INT,
                sender,
                TAG_DONE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            metrics.done_messages++;
            metrics.received_bytes += sizeof(int);

            if (sender >= 1 &&
                sender <= node_count &&
                !finished[sender]) {

                finished[sender] = 1;
                done_count++;
            }

            fprintf(log,
                "DONE node=%d (%d/%d)\n",
                sender,
                done_count,
                node_count
            );

            fflush(log);

        } else {
            fprintf(stderr,
                "[BASE] Unexpected MPI tag %d\n",
                status.MPI_TAG);

            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    /*
     * Termination after all nodes finish.
     *
     * Each charging node sends DONE only
     * when its own enquiries have finished.
     * Finished nodes continue responding
     * to other neighbours until TERMINATE.
     */
    for (int rank = 1;
         rank <= node_count;
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

        metrics.termination_messages++;
        metrics.sent_bytes += sizeof(int);
    }

    double total_time =
        MPI_Wtime() - simulation_start;

    long long messages_received =
        metrics.status_updates +
        metrics.alerts +
        metrics.done_messages;

    long long messages_sent =
        metrics.redirect_counts +
        metrics.redirect_lists +
        metrics.termination_messages;

    fprintf(log,
        "\n========================================\n"
        "SIMULATION SUMMARY\n"
        "========================================\n"
        "Implementation: NONBLOCKING MPI\n"
        "Report history period: %d ms\n"
        "Seed: %u\n"
        "Simulation time: %.9f s\n"
        "EV nodes: %d\n"
        "Completed nodes: %d\n"
        "Status updates: %lld\n"
        "Congestion alerts: %lld\n"
        "Invalid reports: %lld\n"
        "Reports with candidates: %lld\n"
        "Reports without candidates: %lld\n"
        "DONE messages: %lld\n"
        "Redirect count messages: %lld\n"
        "Redirect list messages: %lld\n"
        "Termination messages: %lld\n"
        "Messages received: %lld\n"
        "Messages sent: %lld\n"
        "Total base-station messages: %lld\n"
        "Payload bytes received: %lld\n"
        "Payload bytes sent: %lld\n"
        "Total payload bytes: %lld\n"
        "Total processing time: %.9f s\n"
        "Average processing time per alert: %.9f s\n",
        cfg->report_period_ms,
        cfg->seed,
        total_time,
        node_count,
        done_count,
        metrics.status_updates,
        metrics.alerts,
        metrics.invalid_reports,
        metrics.with_candidates,
        metrics.without_candidates,
        metrics.done_messages,
        metrics.redirect_counts,
        metrics.redirect_lists,
        metrics.termination_messages,
        messages_received,
        messages_sent,
        messages_received + messages_sent,
        metrics.received_bytes,
        metrics.sent_bytes,
        metrics.received_bytes + metrics.sent_bytes,
        metrics.total_processing_time,
        metrics.alerts > 0
            ? metrics.total_processing_time /
                metrics.alerts
            : 0.0
    );

    fclose(log);

    free(statuses);
    free(history);
    free(finished);
    free(candidates);

    printf(
        "[BASE] Alerts=%lld Status=%lld "
        "Completed=%d/%d Time=%.3f s\n",
        metrics.alerts,
        metrics.status_updates,
        done_count,
        node_count,
        total_time
    );

    fflush(stdout);
}

/* =========================================================
 * TASK 1: CONGESTION REPORT HELPER
 * ========================================================= */

static void send_congestion_report(
    int rank,
    const int coords[2],
    const Config *cfg,
    int cycle,
    int local_free,
    int neighbour_count,
    const int neighbours[MAX_NEIGHBOURS],
    const int neighbour_data[MAX_NEIGHBOURS][2],
    double exchange_seconds,
    int *redirect_count,
    MPI_Request *redirect_request
) {
    CongestionReport report;
    memset(&report, 0, sizeof(report));

    report.node_rank = rank;
    report.row = coords[0];
    report.col = coords[1];
    report.cycle = cycle;

    report.total_ports = cfg->ports;
    report.free_ports = local_free;
    report.threshold = cfg->threshold;

    report.neighbour_count = neighbour_count;
    report.neighbour_messages =
        2 * neighbour_count;

    /*
     * One request integer and two response
     * integers per neighbour.
     */
    report.neighbour_bytes =
        3 * neighbour_count * (int)sizeof(int);

    report.timestamp_ms = epoch_ms();
    report.neighbour_exchange_seconds =
        exchange_seconds;

    for (int i = 0; i < MAX_NEIGHBOURS; i++) {

        if (i < neighbour_count) {
            report.neighbour_rank[i] =
                neighbours[i];

            report.neighbour_free[i] =
                neighbour_data[i][0];

            report.neighbour_cycle[i] =
                neighbour_data[i][1];

        } else {
            report.neighbour_rank[i] =
                MPI_PROC_NULL;

            report.neighbour_free[i] = -1;
            report.neighbour_cycle[i] = -1;
        }
    }

    *redirect_count = 0;

    /*
     * Post receive before sending alert.
     */
    MPI_Irecv(
        redirect_count,
        1,
        MPI_INT,
        0,
        TAG_REDIRECT_COUNT,
        MPI_COMM_WORLD,
        redirect_request
    );

    MPI_Send(
        &report,
        (int)sizeof(report),
        MPI_BYTE,
        0,
        TAG_CONGESTION_REPORT,
        MPI_COMM_WORLD
    );
}

/* =========================================================
 * TASK 1: EV CHARGING NODE
 * ========================================================= */

static void run_charging_node(
    int rank,
    MPI_Comm cart_comm,
    const Config *cfg
) {
    int cart_rank;
    int coords[2];

    MPI_Comm_rank(cart_comm, &cart_rank);

    MPI_Cart_coords(
        cart_comm, cart_rank, 2, coords
    );

    int neighbours[MAX_NEIGHBOURS];

    int neighbour_count =
        cart_neighbours(cart_comm, neighbours);

    /*
     * Require one OpenMP thread per charging port.
     * The SLURM job should therefore request:
     *     --cpus-per-task=<ports>
     */
    omp_set_dynamic(0);
    omp_set_num_threads(cfg->ports);

    int actual_threads = 0;

    #pragma omp parallel num_threads(cfg->ports) shared(actual_threads)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    if (actual_threads != cfg->ports) {
        fprintf(
            stderr,
            "Node %d: requested %d OpenMP threads for %d ports, "
            "but runtime created %d threads.\n",
            rank,
            cfg->ports,
            cfg->ports,
            actual_threads
        );

        MPI_Abort(MPI_COMM_WORLD, 22);
    }

    int *ports = malloc(
        (size_t)cfg->ports * sizeof(int)
    );

    unsigned int *seeds = malloc(
        (size_t)cfg->ports *
        sizeof(unsigned int)
    );

    if (!ports || !seeds) {
        MPI_Abort(MPI_COMM_WORLD, 20);
    }

    for (int i = 0; i < cfg->ports; i++) {
        ports[i] = 1;

        seeds[i] =
            cfg->seed
            ^ ((unsigned int)rank * 1009U)
            ^ ((unsigned int)(i + 1) * 7919U);
    }

    char filename[128];

    snprintf(
        filename,
        sizeof(filename),
        "task2_nonblocking_node_%d.log",
        rank
    );

    FILE *log = fopen(filename, "w");

    if (!log) {
        MPI_Abort(MPI_COMM_WORLD, 21);
    }

    fprintf(log,
        "Implementation: NONBLOCKING MPI\n"
        "Deterministic seed: %u\n"
        "OpenMP threads / charging ports: %d\n"
        "World rank: %d\n"
        "Cartesian rank: %d\n"
        "Coordinates: (%d,%d)\n"
        "Neighbour count: %d\n",
        cfg->seed,
        actual_threads,
        rank,
        cart_rank,
        coords[0],
        coords[1],
        neighbour_count
    );

    for (int i = 0; i < neighbour_count; i++) {
        fprintf(log,
            "Neighbour %d = world rank %d\n",
            i, neighbours[i]
        );
    }

    int current_free = cfg->ports;
    int current_cycle = -1;
    int completed_cycles = 0;

    int querying = 0;
    int waiting_redirect = 0;
    int done_sent = 0;
    int terminated = 0;

    int request_values[MAX_NEIGHBOURS];
    int responses[MAX_NEIGHBOURS][2];

    MPI_Request sends[MAX_NEIGHBOURS];
    MPI_Request receives[MAX_NEIGHBOURS];
    MPI_Status send_statuses[MAX_NEIGHBOURS];
    MPI_Status receive_statuses[MAX_NEIGHBOURS];

    MPI_Request redirect_request =
        MPI_REQUEST_NULL;

    int redirect_count = 0;

    int enquiry_cycle = -1;
    int enquiry_free = 0;
    double enquiry_start = 0.0;

    int requests_sent = 0;
    int responses_sent = 0;
    int alerts_sent = 0;
    int status_sent = 0;
    int local_redirections = 0;
    long long completed_queries = 0;

    double total_exchange_time = 0.0;
    double min_exchange_time = DBL_MAX;
    double max_exchange_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    double next_update = MPI_Wtime();

    while (!terminated) {

        int flag = 0;
        MPI_Status status;

        /* --------------------------------------
         * A. TERMINATION MESSAGE
         * -------------------------------------- */
        MPI_Iprobe(
            0, TAG_TERMINATE,
            MPI_COMM_WORLD,
            &flag, &status
        );

        if (flag) {

            int value;

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

        /* --------------------------------------
         * B. RESPOND TO NEIGHBOUR REQUESTS
         * -------------------------------------- */
        MPI_Iprobe(
            MPI_ANY_SOURCE,
            TAG_NEIGHBOUR_REQUEST,
            MPI_COMM_WORLD,
            &flag, &status
        );

        if (flag) {

            int request_value;

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
             * Response contains:
             * [free ports, current cycle]
             */
            int response[2] = {
                current_free,
                current_cycle
            };

            MPI_Send(
                response,
                2,
                MPI_INT,
                status.MPI_SOURCE,
                TAG_NEIGHBOUR_RESPONSE,
                MPI_COMM_WORLD
            );

            responses_sent++;

            fprintf(log,
                "Responded to node %d: "
                "%d free, cycle %d\n",
                status.MPI_SOURCE,
                current_free,
                current_cycle
            );
        }

        /* --------------------------------------
         * C. PROCESS NEIGHBOUR RESPONSES
         * -------------------------------------- */
        if (querying && neighbour_count > 0) {

            int complete = 0;

            MPI_Testall(
                neighbour_count,
                receives,
                &complete,
                receive_statuses
            );

            if (complete) {

                MPI_Waitall(
                    neighbour_count,
                    sends,
                    send_statuses
                );

                double exchange_seconds =
                    MPI_Wtime() - enquiry_start;

                completed_queries++;
                total_exchange_time += exchange_seconds;

                if (exchange_seconds < min_exchange_time)
                    min_exchange_time = exchange_seconds;

                if (exchange_seconds > max_exchange_time)
                    max_exchange_time = exchange_seconds;

                int best_node = -1;
                int best_free = -1;

                for (int i = 0;
                     i < neighbour_count; i++) {

                    int free_count =
                        responses[i][0];

                    if (free_count > cfg->threshold &&
                        free_count > best_free) {

                        best_node = neighbours[i];
                        best_free = free_count;
                    }
                }

                if (best_node >= 0) {

                    local_redirections++;

                    fprintf(log,
                        "LOCAL REDIRECTION\n"
                        "Selected node: %d\n"
                        "Available ports: %d\n"
                        "Exchange time: %.9f s\n",
                        best_node,
                        best_free,
                        exchange_seconds
                    );

                } else {

                    send_congestion_report(
                        rank,
                        coords,
                        cfg,
                        enquiry_cycle,
                        enquiry_free,
                        neighbour_count,
                        neighbours,
                        (const int (*)[2])responses,
                        exchange_seconds,
                        &redirect_count,
                        &redirect_request
                    );

                    waiting_redirect = 1;
                    alerts_sent++;

                    fprintf(log,
                        "CONGESTION ALERT SENT\n"
                        "Free ports: %d/%d\n"
                        "Neighbour messages: %d\n"
                        "Neighbour bytes: %zu\n"
                        "Exchange time: %.9f s\n",
                        enquiry_free,
                        cfg->ports,
                        2 * neighbour_count,
                        3 * (size_t)neighbour_count *
                            sizeof(int),
                        exchange_seconds
                    );
                }

                querying = 0;
            }
        }

        /* --------------------------------------
         * D. BASE STATION RESPONSE
         * -------------------------------------- */
        if (waiting_redirect) {

            int complete = 0;

            MPI_Test(
                &redirect_request,
                &complete,
                MPI_STATUS_IGNORE
            );

            if (complete) {

                int total_nodes =
                    cfg->rows * cfg->cols;

                if (redirect_count < 0 ||
                    redirect_count > total_nodes) {

                    MPI_Abort(MPI_COMM_WORLD, 22);
                }

                fprintf(log,
                    "BASE RESPONSE: %d candidates\n",
                    redirect_count
                );

                if (redirect_count > 0) {

                    int *list = malloc(
                        (size_t)redirect_count *
                        sizeof(int)
                    );

                    if (!list)
                        MPI_Abort(MPI_COMM_WORLD, 23);

                    MPI_Recv(
                        list,
                        redirect_count,
                        MPI_INT,
                        0,
                        TAG_REDIRECT_LIST,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );

                    for (int i = 0;
                         i < redirect_count; i++) {

                        fprintf(log,
                            "Alternative node: %d\n",
                            list[i]
                        );
                    }

                    free(list);

                } else {
                    fprintf(log,
                        "No nearby alternative found.\n");
                }

                waiting_redirect = 0;
            }
        }

        /* --------------------------------------
         * E. PERIODIC PORT UPDATE
         * -------------------------------------- */
        if (completed_cycles < cfg->cycles &&
            !querying &&
            !waiting_redirect &&
            MPI_Wtime() >= next_update) {

            current_cycle = completed_cycles;

            update_ports(
                ports,
                seeds,
                cfg->ports,
                cfg->force_low,
                current_cycle
            );

            current_free = count_free_ports(
                ports, cfg->ports
            );

            long long timestamp = epoch_ms();
            char time_string[80];

            format_time(
                timestamp,
                time_string,
                sizeof(time_string)
            );

            fprintf(log,
                "\n[%s] CYCLE %d\n"
                "Available: %d/%d\n",
                time_string,
                current_cycle,
                current_free,
                cfg->ports
            );

            completed_cycles++;

            next_update =
                MPI_Wtime() +
                cfg->interval_ms / 1000.0;

            /*
             * Send status every cycle so the
             * base station knows about stations
             * that are NOT congested.
             */
            StatusUpdate update = {
                rank,
                current_free,
                current_cycle
            };

            MPI_Send(
                &update,
                (int)sizeof(update),
                MPI_BYTE,
                0,
                TAG_NODE_STATUS_UPDATE,
                MPI_COMM_WORLD
            );

            status_sent++;

            if (current_free <= cfg->threshold) {

                enquiry_cycle = current_cycle;
                enquiry_free = current_free;

                if (neighbour_count > 0) {

                    enquiry_start = MPI_Wtime();

                    for (int i = 0;
                         i < neighbour_count; i++) {

                        responses[i][0] = -1;
                        responses[i][1] = -1;

                        MPI_Irecv(
                            responses[i],
                            2,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_RESPONSE,
                            MPI_COMM_WORLD,
                            &receives[i]
                        );
                    }

                    for (int i = 0;
                         i < neighbour_count; i++) {

                        request_values[i] = rank;

                        MPI_Isend(
                            &request_values[i],
                            1,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_REQUEST,
                            MPI_COMM_WORLD,
                            &sends[i]
                        );

                        requests_sent++;
                    }

                    querying = 1;

                } else {

                    /*
                     * 1x1 grid: no neighbours.
                     */
                    int empty_nodes[4] = {
                        MPI_PROC_NULL,
                        MPI_PROC_NULL,
                        MPI_PROC_NULL,
                        MPI_PROC_NULL
                    };

                    int empty_responses[4][2] = {
                        {-1,-1}, {-1,-1},
                        {-1,-1}, {-1,-1}
                    };

                    send_congestion_report(
                        rank,
                        coords,
                        cfg,
                        current_cycle,
                        current_free,
                        0,
                        empty_nodes,
                        (const int (*)[2])empty_responses,
                        0.0,
                        &redirect_count,
                        &redirect_request
                    );

                    waiting_redirect = 1;
                    alerts_sent++;
                }
            }
        }

        /* --------------------------------------
         * F. DONE / SHUTDOWN HANDSHAKE
         * -------------------------------------- */
        if (completed_cycles >= cfg->cycles &&
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

            fprintf(log,
                "DONE sent; waiting for termination.\n");
        }

        fflush(log);
        sleep_ms(1);
    }

    double average_exchange_time =
        completed_queries > 0
        ? total_exchange_time / (double)completed_queries
        : 0.0;

    if (completed_queries == 0)
        min_exchange_time = 0.0;

    long long neighbour_messages =
        (long long)requests_sent +
        (long long)responses_sent;

    long long neighbour_payload_bytes =
        (long long)requests_sent * (long long)sizeof(int) +
        (long long)responses_sent * 2LL * (long long)sizeof(int);

    long long status_payload_bytes =
        (long long)status_sent *
        (long long)sizeof(StatusUpdate);

    long long report_payload_bytes =
        (long long)alerts_sent *
        (long long)sizeof(CongestionReport);

    fprintf(log,
        "\n========================================\n"
        "NODE SUMMARY\n"
        "========================================\n"
        "Implementation: NONBLOCKING MPI\n"
        "Completed cycles: %d\n"
        "OpenMP threads / charging ports: %d\n"
        "Seed: %u\n"
        "Report history period: %d ms\n"
        "Status updates sent: %d\n"
        "Neighbour requests sent: %d\n"
        "Neighbour responses sent: %d\n"
        "Total neighbour messages: %lld\n"
        "Neighbour payload bytes: %lld\n"
        "Status payload bytes: %lld\n"
        "Congestion report payload bytes: %lld\n"
        "Completed neighbour queries: %lld\n"
        "Local redirections: %d\n"
        "Congestion reports sent: %d\n"
        "Total neighbour exchange time: %.9f s\n"
        "Average neighbour exchange time: %.9f s\n"
        "Minimum neighbour exchange time: %.9f s\n"
        "Maximum neighbour exchange time: %.9f s\n"
        "Graceful shutdown completed.\n",
        completed_cycles,
        actual_threads,
        cfg->seed,
        cfg->report_period_ms,
        status_sent,
        requests_sent,
        responses_sent,
        neighbour_messages,
        neighbour_payload_bytes,
        status_payload_bytes,
        report_payload_bytes,
        completed_queries,
        local_redirections,
        alerts_sent,
        total_exchange_time,
        average_exchange_time,
        min_exchange_time,
        max_exchange_time
    );

    fclose(log);

    free(ports);
    free(seeds);

    printf(
        "[NODE %d] Cycles=%d Alerts=%d\n",
        rank,
        completed_cycles,
        alerts_sent
    );

    fflush(stdout);
}

/* =========================================================
 * MAIN
 * ========================================================= */

int main(int argc, char **argv)
{
    int provided = 0;

    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {

        if (rank == 0)
            fprintf(stderr,
                "MPI_THREAD_FUNNELED unavailable\n");

        MPI_Abort(MPI_COMM_WORLD, 30);
    }

    if (argc < 3 || argc > 10) {

        if (rank == 0) {
            fprintf(stderr,
                "Usage: %s rows cols "
                "[ports] [cycles] [interval_ms] "
                "[threshold] [force_low] "
                "[report_period_ms] [seed]\n",
                argv[0]
            );
        }

        MPI_Finalize();
        return 1;
    }

    Config cfg;

    cfg.rows = atoi(argv[1]);
    cfg.cols = atoi(argv[2]);

    cfg.ports =
        argc > 3 ? atoi(argv[3]) :
        DEFAULT_PORTS;

    cfg.cycles =
        argc > 4 ? atoi(argv[4]) :
        DEFAULT_CYCLES;

    cfg.interval_ms =
        argc > 5 ? atoi(argv[5]) :
        DEFAULT_INTERVAL_MS;

    cfg.threshold =
        argc > 6 ? atoi(argv[6]) :
        DEFAULT_THRESHOLD;

    cfg.force_low =
        argc > 7 ? (atoi(argv[7]) != 0) : 0;

    cfg.report_period_ms =
        argc > 8 ? atoi(argv[8]) :
        DEFAULT_REPORT_PERIOD_MS;

    cfg.seed =
        argc > 9
        ? (unsigned int)strtoul(argv[9], NULL, 10)
        : DEFAULT_SEED;

    long long total_nodes =
        (long long)cfg.rows * cfg.cols;

    if (cfg.rows <= 0 ||
        cfg.cols <= 0 ||
        cfg.ports <= 0 ||
        cfg.cycles <= 0 ||
        cfg.interval_ms <= 0 ||
        cfg.threshold < 0 ||
        cfg.threshold >= cfg.ports ||
        cfg.report_period_ms <= 0 ||
        total_nodes > INT_MAX - 1 ||
        total_nodes + 1 != size) {

        if (rank == 0)
            fprintf(stderr,
                "Invalid configuration.\n"
                "Required processes = "
                "rows * cols + 1\n");

        MPI_Finalize();
        return 1;
    }

    /*
     * Isolate EV nodes from the base station.
     */
    MPI_Comm worker_comm = MPI_COMM_NULL;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        rank == 0 ? MPI_UNDEFINED : 1,
        rank,
        &worker_comm
    );

    if (rank == 0) {

        MPI_Barrier(MPI_COMM_WORLD);

        run_base_station(&cfg);

    } else {

        MPI_Comm cart_comm = MPI_COMM_NULL;

        int dimensions[2] = {
            cfg.rows,
            cfg.cols
        };

        int periods[2] = {0, 0};

        MPI_Cart_create(
            worker_comm,
            2,
            dimensions,
            periods,
            0,
            &cart_comm
        );

        if (cart_comm == MPI_COMM_NULL)
            MPI_Abort(MPI_COMM_WORLD, 31);

        run_charging_node(
            rank,
            cart_comm,
            &cfg
        );

        MPI_Comm_free(&cart_comm);
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();

    return 0;
}
