
#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#define MAX_NEIGHBOURS 4
#define DEFAULT_PORTS 6
#define DEFAULT_CYCLES 8
#define DEFAULT_INTERVAL_MS 1000
#define DEFAULT_THRESHOLD 1

/*
 * Base station: MPI_COMM_WORLD rank 0
 * EV nodes:     MPI_COMM_WORLD ranks 1..N
 *
 * Cartesian rank = world rank - 1
 * Cartesian reorder = 0
 *
 * Only the main thread makes MPI calls.
 */

enum {
    TAG_NEIGHBOUR_REQUEST = 100,
    TAG_NEIGHBOUR_RESPONSE = 101,
    TAG_CONGESTION_REPORT = 200,
    TAG_REDIRECT = 201,
    TAG_DONE = 998,
    TAG_TERMINATE = 999
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

static long long now_epoch_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    return (long long)ts.tv_sec * 1000LL
         + ts.tv_nsec / 1000000LL;
}

static void format_time(
    long long epoch_ms,
    char *buffer,
    size_t size
) {
    time_t sec = (time_t)(epoch_ms / 1000LL);
    struct tm t;
    char base[64];

    localtime_r(&sec, &t);

    strftime(base, sizeof(base),
             "%Y-%m-%d %H:%M:%S", &t);

    snprintf(buffer, size, "%s.%03lld",
             base, epoch_ms % 1000LL);
}

static void sleep_ms(int ms)
{
    if (ms <= 0) return;

    struct timespec req;
    req.tv_sec = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;

    while (nanosleep(&req, &req) != 0) {
        /* Retry interrupted sleep. */
    }
}

/*
 * OpenMP charging-port simulation.
 *
 * 1 = free
 * 0 = occupied
 *
 * Each port has its own random seed.
 */
static void update_ports(
    int *port_status,
    unsigned int *seeds,
    int ports,
    int force_low,
    int cycle
) {
    int free_probability = force_low ? 10 : 55;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ports; i++) {

        if (force_low && cycle == 0) {
            /* Deterministic congestion test. */
            port_status[i] = 0;
        } else {
            int value =
                (int)(rand_r(&seeds[i]) % 100U);

            port_status[i] =
                (value < free_probability) ? 1 : 0;
        }
    }
}

/*
 * Count available ports using OpenMP reduction.
 */
static int count_available(
    const int *port_status,
    int ports
) {
    int total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < ports; i++) {
        total += port_status[i];
    }

    return total;
}

/*
 * Translate a Cartesian rank into a world rank.
 *
 * Because reorder=0 and EV nodes occupy
 * world ranks 1..N:
 *
 * world rank = cart rank + 1
 */
static int cart_to_world(int cart_rank)
{
    if (cart_rank == MPI_PROC_NULL)
        return MPI_PROC_NULL;

    return cart_rank + 1;
}

static int discover_neighbours(
    MPI_Comm cart_comm,
    int neighbours[MAX_NEIGHBOURS]
) {
    int top, bottom, left, right;

    MPI_Cart_shift(
        cart_comm, 0, 1, &top, &bottom
    );

    MPI_Cart_shift(
        cart_comm, 1, 1, &left, &right
    );

    int cart_neighbours[4] = {
        top, bottom, left, right
    };

    int count = 0;

    for (int i = 0; i < 4; i++) {
        if (cart_neighbours[i] != MPI_PROC_NULL) {
            neighbours[count++] =
                cart_to_world(cart_neighbours[i]);
        }
    }

    for (int i = count; i < 4; i++) {
        neighbours[i] = MPI_PROC_NULL;
    }

    return count;
}

/*
 * Temporary Task 1 base-station test harness.
 *
 * This does not implement the full Task 2
 * nearest-available-node search.
 *
 * It receives congestion reports and returns
 * an empty candidate list.
 *
 * It terminates nodes only after all nodes
 * have completed their scheduled cycles and
 * outstanding enquiries.
 */
static void run_base_station(int workers)
{
    FILE *log =
        fopen("task1_base_station.log", "w");

    if (!log) {
        fprintf(stderr,
                "Base station: log open failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int done_count = 0;
    int alerts = 0;
    long long report_bytes = 0;

    double start = MPI_Wtime();

    fprintf(log,
            "TASK 1 BASE STATION TEST HARNESS\n");
    fprintf(log, "EV nodes: %d\n", workers);
    fflush(log);

    while (done_count < workers) {

        MPI_Status status;
        int flag = 0;

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

        if (status.MPI_TAG == TAG_CONGESTION_REPORT) {

            CongestionReport report;
            memset(&report, 0, sizeof(report));

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
            report_bytes += sizeof(report);

            char timestamp[80];

            format_time(
                report.epoch_ms,
                timestamp,
                sizeof(timestamp)
            );

            fprintf(log,
                "\n[%s] CONGESTION ALERT\n"
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
                 i < report.valid_neighbour_count;
                 i++) {

                fprintf(log,
                    "Neighbour %d: %d available\n",
                    report.neighbour_ranks[i],
                    report.neighbour_available[i]
                );
            }

            /*
             * Task 2 replaces this placeholder.
             */
            RedirectReply reply;
            memset(&reply, 0, sizeof(reply));

            for (int i = 0; i < MAX_NEIGHBOURS; i++)
                reply.candidate_ranks[i] = -1;

            MPI_Send(
                &reply,
                (int)sizeof(reply),
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_REDIRECT,
                MPI_COMM_WORLD
            );

            fprintf(log,
                    "Placeholder response sent.\n");
            fflush(log);

        } else if (status.MPI_TAG == TAG_DONE) {

            int value;

            MPI_Recv(
                &value, 1, MPI_INT,
                status.MPI_SOURCE,
                TAG_DONE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            done_count++;

            fprintf(log,
                "Node %d completed (%d/%d)\n",
                status.MPI_SOURCE,
                done_count,
                workers
            );

            fflush(log);
        }
    }

    /*
     * Every node has completed its own
     * requests and base-station responses.
     * Release the entire network.
     */
    for (int rank = 1; rank <= workers; rank++) {
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

    double total = MPI_Wtime() - start;

    fprintf(log,
        "\nSIMULATION SUMMARY\n"
        "Simulation time: %.6f s\n"
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
        "[BASE] Completed. Alerts=%d Time=%.3fs\n",
        alerts, total
    );
    fflush(stdout);
}

/*
 * Execute one charging-node process.
 */
static void run_charging_node(
    int world_rank,
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
        discover_neighbours(cart_comm, neighbours);

    /*
     * Shared local charging-port arrays.
     */
    int *port_status =
        malloc((size_t)cfg->ports * sizeof(int));

    unsigned int *seeds =
        malloc((size_t)cfg->ports *
               sizeof(unsigned int));

    if (!port_status || !seeds) {
        fprintf(stderr,
                "Node %d: allocation failed\n",
                world_rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    for (int i = 0; i < cfg->ports; i++) {
        port_status[i] = 1;

        seeds[i] =
            (unsigned int)time(NULL)
            ^ ((unsigned int)world_rank * 1009U)
            ^ ((unsigned int)(i + 1) * 7919U);
    }

    char filename[128];

    snprintf(
        filename, sizeof(filename),
        "task1_node_%d.log", world_rank
    );

    FILE *log = fopen(filename, "w");

    if (!log) {
        fprintf(stderr,
                "Node %d: log open failed\n",
                world_rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    fprintf(log,
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

    for (int i = 0; i < neighbour_count; i++) {
        fprintf(log,
                "Neighbour %d = world rank %d\n",
                i, neighbours[i]);
    }

    fflush(log);

    /*
     * Event-driven communication state.
     *
     * Each node may have one outstanding
     * availability enquiry at a time.
     */
    int querying = 0;
    int waiting_redirect = 0;

    int response_values[MAX_NEIGHBOURS];
    int request_payload[MAX_NEIGHBOURS];

    MPI_Request request_sends[MAX_NEIGHBOURS];
    MPI_Request response_receives[MAX_NEIGHBOURS];

    MPI_Request redirect_recv = MPI_REQUEST_NULL;

    RedirectReply redirect_reply;

    int enquiry_free_ports = 0;
    double enquiry_start = 0.0;

    int completed_cycles = 0;
    int current_available = cfg->ports;

    int done_sent = 0;
    int terminated = 0;

    int total_requests_sent = 0;
    int total_responses_sent = 0;
    int total_alerts = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    double next_update = MPI_Wtime();

    while (!terminated) {

        /*
         * A. Check whether the base station
         *    has authorized termination.
         */
        int terminate_flag = 0;
        MPI_Status terminate_status;

        MPI_Iprobe(
            0, TAG_TERMINATE,
            MPI_COMM_WORLD,
            &terminate_flag,
            &terminate_status
        );

        if (terminate_flag) {
            int value;

            MPI_Recv(
                &value, 1, MPI_INT,
                0, TAG_TERMINATE,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            terminated = 1;
            break;
        }

        /*
         * B. Respond to incoming neighbour
         *    requests, even while this node
         *    is waiting for its own responses.
         *
         * The requester posts its Irecv
         * before sending its request.
         */
        int incoming = 0;
        MPI_Status request_status;

        MPI_Iprobe(
            MPI_ANY_SOURCE,
            TAG_NEIGHBOUR_REQUEST,
            MPI_COMM_WORLD,
            &incoming,
            &request_status
        );

        if (incoming) {
            int request_value;

            MPI_Recv(
                &request_value,
                1,
                MPI_INT,
                request_status.MPI_SOURCE,
                TAG_NEIGHBOUR_REQUEST,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            int available = current_available;

            MPI_Send(
                &available,
                1,
                MPI_INT,
                request_status.MPI_SOURCE,
                TAG_NEIGHBOUR_RESPONSE,
                MPI_COMM_WORLD
            );

            total_responses_sent++;

            fprintf(log,
                "Responded to node %d: %d free\n",
                request_status.MPI_SOURCE,
                available
            );
        }

        /*
         * C. Check whether all neighbour
         *    responses have arrived.
         */
        if (querying) {
            int complete = 0;

            MPI_Testall(
                neighbour_count,
                response_receives,
                &complete,
                MPI_STATUSES_IGNORE
            );

            if (complete) {
                MPI_Waitall(
                    neighbour_count,
                    request_sends,
                    MPI_STATUSES_IGNORE
                );

                double elapsed =
                    MPI_Wtime() - enquiry_start;

                int best_node = -1;
                int best_available = -1;

                for (int i = 0;
                     i < neighbour_count;
                     i++) {

                    int available =
                        response_values[i];

                    if (available > cfg->threshold &&
                        available > best_available) {

                        best_available = available;
                        best_node = neighbours[i];
                    }
                }

                if (best_node >= 0) {

                    fprintf(log,
                        "LOCAL REDIRECTION\n"
                        "Selected node: %d\n"
                        "Available ports: %d\n"
                        "Exchange time: %.9f s\n",
                        best_node,
                        best_available,
                        elapsed
                    );

                } else {

                    /*
                     * No neighbour satisfies
                     * the availability threshold.
                     */
                    CongestionReport report;
                    memset(&report, 0,
                           sizeof(report));

                    report.node_rank = world_rank;
                    report.coord_row = coords[0];
                    report.coord_col = coords[1];

                    report.total_ports = cfg->ports;
                    report.local_available =
                        enquiry_free_ports;
                    report.threshold = cfg->threshold;

                    report.epoch_ms = now_epoch_ms();

                    report.valid_neighbour_count =
                        neighbour_count;

                    report.neighbour_messages =
                        2 * neighbour_count;

                    report.neighbour_bytes =
                        2 * neighbour_count *
                        (int)sizeof(int);

                    report.neighbour_exchange_seconds =
                        elapsed;

                    for (int i = 0;
                         i < MAX_NEIGHBOURS;
                         i++) {

                        if (i < neighbour_count) {
                            report.neighbour_ranks[i] =
                                neighbours[i];

                            report.neighbour_available[i] =
                                response_values[i];
                        } else {
                            report.neighbour_ranks[i] =
                                MPI_PROC_NULL;

                            report.neighbour_available[i] =
                                -1;
                        }
                    }

                    memset(
                        &redirect_reply, 0,
                        sizeof(redirect_reply)
                    );

                    /*
                     * Post reply receive before
                     * submitting the report.
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

                    waiting_redirect = 1;

                    MPI_Send(
                        &report,
                        (int)sizeof(report),
                        MPI_BYTE,
                        0,
                        TAG_CONGESTION_REPORT,
                        MPI_COMM_WORLD
                    );

                    total_alerts++;

                    fprintf(log,
                        "CONGESTION ALERT SENT\n"
                        "Free ports: %d/%d\n"
                        "Neighbour messages: %d\n"
                        "Neighbour bytes: %d\n"
                        "Exchange time: %.9f s\n",
                        report.local_available,
                        report.total_ports,
                        report.neighbour_messages,
                        report.neighbour_bytes,
                        elapsed
                    );
                }

                querying = 0;
            }
        }

        /*
         * D. Process base-station response.
         */
        if (waiting_redirect) {

            int complete = 0;

            MPI_Test(
                &redirect_recv,
                &complete,
                MPI_STATUS_IGNORE
            );

            if (complete) {

                fprintf(log,
                    "BASE STATION RESPONSE: "
                    "%d candidates\n",
                    redirect_reply.candidate_count
                );

                for (int i = 0;
                     i < redirect_reply.candidate_count &&
                     i < MAX_NEIGHBOURS;
                     i++) {

                    fprintf(log,
                        "Candidate: %d\n",
                        redirect_reply.candidate_ranks[i]
                    );
                }

                waiting_redirect = 0;
            }
        }

        /*
         * E. Perform the next charging-port
         *    update when no own enquiry is
         *    outstanding.
         *
         * Incoming requests are still
         * serviced during the waiting period.
         */
        double now = MPI_Wtime();

        if (completed_cycles < cfg->cycles &&
            !querying &&
            !waiting_redirect &&
            now >= next_update) {

            update_ports(
                port_status,
                seeds,
                cfg->ports,
                cfg->force_low,
                completed_cycles
            );

            current_available =
                count_available(
                    port_status,
                    cfg->ports
                );

            long long timestamp = now_epoch_ms();

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
                completed_cycles,
                current_available,
                cfg->ports
            );

            completed_cycles++;

            next_update =
                MPI_Wtime() +
                cfg->interval_ms / 1000.0;

            /*
             * Request availability only
             * when the station is congested.
             */
            if (current_available <=
                cfg->threshold) {

                if (neighbour_count > 0) {

                    enquiry_free_ports =
                        current_available;

                    enquiry_start = MPI_Wtime();

                    for (int i = 0;
                         i < neighbour_count;
                         i++) {

                        response_values[i] = -1;

                        /*
                         * Receive posted before
                         * sending each request.
                         */
                        MPI_Irecv(
                            &response_values[i],
                            1,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_RESPONSE,
                            MPI_COMM_WORLD,
                            &response_receives[i]
                        );
                    }

                    for (int i = 0;
                         i < neighbour_count;
                         i++) {

                        request_payload[i] =
                            world_rank;

                        MPI_Isend(
                            &request_payload[i],
                            1,
                            MPI_INT,
                            neighbours[i],
                            TAG_NEIGHBOUR_REQUEST,
                            MPI_COMM_WORLD,
                            &request_sends[i]
                        );

                        total_requests_sent++;
                    }

                    querying = 1;

                } else {
                    CongestionReport report;
                    memset(&report, 0, sizeof(report));

                    report.node_rank = world_rank;
                    report.coord_row = coords[0];
                    report.coord_col = coords[1];
                    report.total_ports = cfg->ports;
                    report.local_available = current_available;
                    report.threshold = cfg->threshold;
                    report.epoch_ms = timestamp;
                    report.valid_neighbour_count = 0;

                    for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                        report.neighbour_ranks[i] = MPI_PROC_NULL;
                        report.neighbour_available[i] = -1;
                    }

                    memset(&redirect_reply, 0, sizeof(redirect_reply));

                    MPI_Irecv(
                        &redirect_reply,
                        (int)sizeof(redirect_reply),
                        MPI_BYTE,
                        0, TAG_REDIRECT,
                        MPI_COMM_WORLD,
                        &redirect_recv
                    );

                    waiting_redirect = 1;

                    MPI_Send(
                        &report,
                        (int)sizeof(report),
                        MPI_BYTE,
                        0, TAG_CONGESTION_REPORT,
                        MPI_COMM_WORLD
                    );

                    total_alerts++;
                }
            }
        }

        /*
         * F. Finish own work.
         *
         * A completed node continues
         * responding to other nodes until
         * the base station terminates it.
         */
        if (completed_cycles >= cfg->cycles &&
            !querying &&
            !waiting_redirect &&
            !done_sent) {

            int done = 1;

            MPI_Send(
                &done, 1, MPI_INT,
                0, TAG_DONE,
                MPI_COMM_WORLD
            );

            done_sent = 1;

            fprintf(log,
                "All local cycles completed.\n"
                "Waiting for termination.\n"
            );
        }

        fflush(log);
        sleep_ms(1);
    }

    fprintf(log,
        "\nNODE SUMMARY\n"
        "Cycles: %d\n"
        "Neighbour requests sent: %d\n"
        "Neighbour responses sent: %d\n"
        "Congestion reports sent: %d\n"
        "Graceful shutdown completed.\n",
        completed_cycles,
        total_requests_sent,
        total_responses_sent,
        total_alerts
    );

    fclose(log);
    free(port_status);
    free(seeds);

    printf(
        "[NODE %d] Completed %d cycles; alerts=%d\n",
        world_rank,
        completed_cycles,
        total_alerts
    );
    fflush(stdout);
}

int main(int argc, char **argv)
{
    int provided = 0;

    MPI_Init_thread(
        &argc, &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr,
                "MPI_THREAD_FUNNELED unavailable\n");
        }

        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    if (argc < 3 || argc > 8) {
        if (rank == 0) {
            fprintf(stderr,
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

    long long nodes =
        (long long)cfg.rows * cfg.cols;

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
            fprintf(stderr,
                "Invalid configuration.\n"
                "Expected MPI processes = "
                "rows * cols + 1\n");
        }

        MPI_Finalize();
        return 1;
    }

    /*
     * Rank 0 = base station.
     * Ranks 1..N = EV charging nodes.
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

        run_base_station((int)nodes);

    } else {

        MPI_Comm cart_comm = MPI_COMM_NULL;

        int dims[2] = {
            cfg.rows,
            cfg.cols
        };

        int periods[2] = {0, 0};

        /*
         * Reorder disabled to guarantee
         * cart_rank = world_rank - 1.
         */
        MPI_Cart_create(
            worker_comm,
            2,
            dims,
            periods,
            0,
            &cart_comm
        );

        if (cart_comm == MPI_COMM_NULL) {
            fprintf(stderr,
                "Node %d: Cartesian creation failed\n",
                rank);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }

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
