
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define MAX_NEIGHBOURS 4

// MPI message tags
enum {
    TAG_REQUEST = 10,
    TAG_RESPONSE = 11,
    TAG_ALERT = 12,
    TAG_REDIRECT = 13,
    TAG_STOP = 14,
    TAG_READY = 15,
    TAG_EXIT = 16
};

// Congestion report sent to base station
typedef struct {
    int node_id;
    int row, col;
    int total_ports;
    int free_ports;
    int neighbour_count;
    int neighbour_id[MAX_NEIGHBOURS];
    int neighbour_free[MAX_NEIGHBOURS];
    int messages_exchanged;
    int message_bytes;
    long long timestamp;
} CongestionReport;

// Configuration supplied by user
typedef struct {
    int rows;
    int cols;
    int ports;
    int threshold;
    int force_congestion;
    double duration;
    double interval;
} Config;

static void short_sleep(void) {
    struct timespec ts = {0, 1000000L};
    nanosleep(&ts, NULL);
}

// ---------------------------------------------------------
// OpenMP charging-port simulation
// 0 = FREE, 1 = IN USE
// ---------------------------------------------------------
static int update_ports(
    int *ports,
    unsigned int *seeds,
    int k,
    int cycle,
    int force_congestion
) {
    // Each iteration updates a separate array element.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k; i++) {
        if (force_congestion && cycle == 0) {
            ports[i] = 1;
        } else {
            // 80% probability that a port is in use.
            ports[i] = (rand_r(&seeds[i]) % 100 < 80)
                       ? 1 : 0;
        }
    }

    // The implicit OpenMP barrier ensures updates
    // have finished before the array is read.
    int free_count = 0;

    #pragma omp parallel for reduction(+:free_count)
    for (int i = 0; i < k; i++) {
        if (ports[i] == 0) {
            free_count++;
        }
    }

    return free_count;
}

// ---------------------------------------------------------
// Charging node
// ---------------------------------------------------------
static void run_charging_node(
    int world_rank,
    MPI_Comm cart_comm,
    const Config *cfg
) {
    int cart_rank;
    int coords[2];

    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    // Find immediate neighbouring nodes.
    int top, bottom, left, right;

    MPI_Cart_shift(cart_comm, 0, 1, &top, &bottom);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    int candidates[4] = {top, bottom, left, right};
    int neighbours[4];
    int neighbour_count = 0;

    for (int i = 0; i < 4; i++) {
        if (candidates[i] != MPI_PROC_NULL) {
            neighbours[neighbour_count++] = candidates[i];
        }
    }

    // Each node owns its local shared-memory arrays.
    int *ports = malloc(cfg->ports * sizeof(int));
    unsigned int *seeds =
        malloc(cfg->ports * sizeof(unsigned int));

    if (!ports || !seeds) {
        fprintf(stderr, "Rank %d: allocation failed\n",
                world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < cfg->ports; i++) {
        ports[i] = 0;
        seeds[i] = (unsigned int)time(NULL)
                 ^ ((unsigned int)(world_rank + 1) * 7919u)
                 ^ ((unsigned int)(i + 1) * 104729u);
    }

    // State for one outstanding neighbour enquiry.
    int querying = 0;
    int request_data[4];
    int responses[4][2];
    MPI_Request send_requests[4];
    MPI_Request recv_requests[4];

    // State for one outstanding base-station report.
    int waiting_redirect = 0;
    int redirect_target = -1;
    MPI_Request redirect_request = MPI_REQUEST_NULL;

    int stopping = 0;
    int ready_sent = 0;
    int exiting = 0;

    int cycle = 0;
    int free_ports = cfg->ports;
    double next_update = 0.0;

    printf("[NODE %d] Coordinates=(%d,%d), neighbours=%d\n",
           world_rank, coords[0], coords[1],
           neighbour_count);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    next_update = start;

    while (!exiting) {
        MPI_Status status;
        int flag;

        // -------------------------------------------------
        // A. Handle incoming MPI messages
        // -------------------------------------------------
        do {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG,
                       MPI_COMM_WORLD, &flag, &status);

            if (!flag) break;

            if (status.MPI_TAG == TAG_REQUEST) {
                int requester;

                MPI_Recv(&requester, 1, MPI_INT,
                         status.MPI_SOURCE, TAG_REQUEST,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Reply with current local availability.
                int reply[2] = {free_ports, cfg->ports};

                MPI_Send(reply, 2, MPI_INT,
                         status.MPI_SOURCE, TAG_RESPONSE,
                         MPI_COMM_WORLD);

            } else if (status.MPI_TAG == TAG_STOP) {
                int dummy;

                MPI_Recv(&dummy, 1, MPI_INT,
                         status.MPI_SOURCE, TAG_STOP,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // No new enquiries, but finish existing ones.
                stopping = 1;

            } else if (status.MPI_TAG == TAG_EXIT) {
                int dummy;

                MPI_Recv(&dummy, 1, MPI_INT,
                         status.MPI_SOURCE, TAG_EXIT,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                exiting = 1;

            } else {
                // Responses and redirections have matching
                // nonblocking receives, so leave them alone.
                break;
            }

        } while (!exiting);

        if (exiting) break;

        // -------------------------------------------------
        // B. Check neighbour enquiry completion
        // -------------------------------------------------
        if (querying) {
            int completed = 0;

            MPI_Testall(neighbour_count, recv_requests,
                        &completed, MPI_STATUSES_IGNORE);

            if (completed) {
                // Ensure outgoing request buffers are safe
                // to reuse.
                MPI_Waitall(neighbour_count, send_requests,
                            MPI_STATUSES_IGNORE);

                int all_congested = 1;
                int best_neighbour = -1;
                int best_free = -1;

                for (int i = 0; i < neighbour_count; i++) {
                    int available = responses[i][0];

                    if (available > cfg->threshold) {
                        all_congested = 0;

                        if (available > best_free) {
                            best_free = available;
                            best_neighbour = neighbours[i];
                        }
                    }
                }

                if (!all_congested) {
                    printf("[NODE %d] Redirect locally to "
                           "node %d (%d free ports)\n",
                           world_rank, best_neighbour,
                           best_free);

                } else {
                    // -------------------------------------
                    // C. Build a detailed congestion report
                    // -------------------------------------
                    CongestionReport report = {0};

                    report.node_id = world_rank;
                    report.row = coords[0];
                    report.col = coords[1];
                    report.total_ports = cfg->ports;
                    report.free_ports = free_ports;
                    report.neighbour_count = neighbour_count;
                    report.timestamp = (long long)time(NULL);

                    // One request + one response per neighbour.
                    report.messages_exchanged =
                        2 * neighbour_count;

                    report.message_bytes =
                        neighbour_count *
                        (sizeof(int) + 2 * sizeof(int));

                    for (int i = 0; i < neighbour_count; i++) {
                        report.neighbour_id[i] = neighbours[i];
                        report.neighbour_free[i] = responses[i][0];
                    }

                    // Post receive before sending report.
                    redirect_target = -1;

                    MPI_Irecv(&redirect_target, 1, MPI_INT,
                              cfg->rows * cfg->cols,
                              TAG_REDIRECT, MPI_COMM_WORLD,
                              &redirect_request);

                    waiting_redirect = 1;

                    MPI_Send(&report, sizeof(report), MPI_BYTE,
                             cfg->rows * cfg->cols,
                             TAG_ALERT, MPI_COMM_WORLD);

                    printf("[NODE %d] Congestion reported "
                           "to base station\n", world_rank);
                }

                fflush(stdout);
                querying = 0;
            }
        }

        // -------------------------------------------------
        // D. Check base-station response
        // -------------------------------------------------
        if (waiting_redirect) {
            int completed = 0;

            MPI_Test(&redirect_request, &completed,
                     MPI_STATUS_IGNORE);

            if (completed) {
                if (redirect_target >= 0) {
                    printf("[NODE %d] Base station suggests "
                           "node %d\n",
                           world_rank, redirect_target);
                } else {
                    printf("[NODE %d] No alternative returned "
                           "by temporary base station\n",
                           world_rank);
                }

                fflush(stdout);
                waiting_redirect = 0;
            }
        }

        // -------------------------------------------------
        // E. Periodically update local charging ports
        // -------------------------------------------------
        double now = MPI_Wtime();

        if (!stopping &&
            !querying &&
            !waiting_redirect &&
            now >= next_update) {

            free_ports = update_ports(
                ports, seeds, cfg->ports, cycle,
                cfg->force_congestion
            );

            printf("[NODE %d] Cycle=%d Free=%d/%d\n",
                   world_rank, cycle, free_ports, cfg->ports);
            fflush(stdout);

            cycle++;
            next_update = now + cfg->interval;

            // Enquire only when locally congested.
            if (free_ports <= cfg->threshold &&
                neighbour_count > 0) {

                // Post all response receives first.
                for (int i = 0; i < neighbour_count; i++) {
                    MPI_Irecv(
                        responses[i], 2, MPI_INT,
                        neighbours[i], TAG_RESPONSE,
                        MPI_COMM_WORLD, &recv_requests[i]
                    );
                }

                // Nonblocking requests to all neighbours.
                for (int i = 0; i < neighbour_count; i++) {
                    request_data[i] = world_rank;

                    MPI_Isend(
                        &request_data[i], 1, MPI_INT,
                        neighbours[i], TAG_REQUEST,
                        MPI_COMM_WORLD, &send_requests[i]
                    );
                }

                querying = 1;
            }
        }

        // -------------------------------------------------
        // F. Two-phase graceful shutdown
        // -------------------------------------------------
        if (stopping && !querying &&
            !waiting_redirect && !ready_sent) {

            int ready = 1;

            MPI_Send(&ready, 1, MPI_INT,
                     cfg->rows * cfg->cols,
                     TAG_READY, MPI_COMM_WORLD);

            ready_sent = 1;
        }

        short_sleep();
    }

    // All active MPI operations have completed.
    free(ports);
    free(seeds);

    MPI_Comm_free(&cart_comm);

    printf("[NODE %d] Graceful shutdown completed\n",
           world_rank);
    fflush(stdout);
}

// ---------------------------------------------------------
// Temporary base station for Task 1 testing
// NOT the final Task 2 implementation.
// ---------------------------------------------------------
static void run_base_station(const Config *cfg) {
    int nodes = cfg->rows * cfg->cols;

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    int stop_sent = 0;
    int ready_count = 0;
    int alerts = 0;

    while (ready_count < nodes) {
        double now = MPI_Wtime();

        // Stop the simulation after its duration.
        if (!stop_sent && now - start >= cfg->duration) {
            for (int r = 0; r < nodes; r++) {
                int dummy = 1;

                MPI_Send(&dummy, 1, MPI_INT,
                         r, TAG_STOP, MPI_COMM_WORLD);
            }

            stop_sent = 1;
        }

        int flag;
        MPI_Status status;

        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG,
                   MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            if (status.MPI_TAG == TAG_ALERT) {
                CongestionReport report;

                MPI_Recv(&report, sizeof(report), MPI_BYTE,
                         status.MPI_SOURCE, TAG_ALERT,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                alerts++;

                printf("[BASE] Alert %d from node %d "
                       "(%d,%d), free=%d/%d, "
                       "neighbour messages=%d, bytes=%d\n",
                       alerts,
                       report.node_id,
                       report.row,
                       report.col,
                       report.free_ports,
                       report.total_ports,
                       report.messages_exchanged,
                       report.message_bytes);

                for (int i = 0; i < report.neighbour_count; i++) {
                    printf("    Neighbour %d: %d free\n",
                           report.neighbour_id[i],
                           report.neighbour_free[i]);
                }

                // Placeholder response only.
                // Task 2 will calculate actual candidates.
                int target = -1;

                MPI_Send(&target, 1, MPI_INT,
                         status.MPI_SOURCE, TAG_REDIRECT,
                         MPI_COMM_WORLD);

                fflush(stdout);

            } else if (status.MPI_TAG == TAG_READY) {
                int ready;

                MPI_Recv(&ready, 1, MPI_INT,
                         status.MPI_SOURCE, TAG_READY,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                ready_count++;
            }

        } else {
            short_sleep();
        }
    }

    // Every node has completed outstanding enquiries.
    // It is now safe to release all processes.
    for (int r = 0; r < nodes; r++) {
        int dummy = 1;

        MPI_Send(&dummy, 1, MPI_INT,
                 r, TAG_EXIT, MPI_COMM_WORLD);
    }

    printf("[BASE] Simulation finished. Total alerts=%d\n",
           alerts);
    fflush(stdout);
}

// ---------------------------------------------------------
// Main program
// ---------------------------------------------------------
int main(int argc, char **argv) {
    int provided;

    MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &provided
    );

    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Insufficient MPI thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 || argc > 8) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: %s rows cols [ports] [seconds] "
                "[threshold] [interval] [force]\n",
                argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    Config cfg;

    cfg.rows = atoi(argv[1]);
    cfg.cols = atoi(argv[2]);
    cfg.ports = argc > 3 ? atoi(argv[3]) : 6;
    cfg.duration = argc > 4 ? atof(argv[4]) : 20.0;
    cfg.threshold = argc > 5 ? atoi(argv[5]) : 1;
    cfg.interval = argc > 6 ? atof(argv[6]) : 2.0;
    cfg.force_congestion = argc > 7 ? atoi(argv[7]) : 0;

    int nodes = cfg.rows * cfg.cols;

    if (cfg.rows <= 0 || cfg.cols <= 0 ||
        cfg.ports <= 0 ||
        cfg.duration <= 0 ||
        cfg.interval <= 0 ||
        cfg.threshold < 0 ||
        cfg.threshold >= cfg.ports ||
        (cfg.force_congestion != 0 &&
         cfg.force_congestion != 1) ||
        nodes + 1 != size) {

        if (rank == 0) {
            fprintf(stderr, "Invalid configuration.\n");
            fprintf(stderr,
                    "Required MPI processes = rows * cols + 1\n");
        }

        MPI_Finalize();
        return 1;
    }

    int base_rank = nodes;

    // Isolate charging nodes from the base station.
    MPI_Comm node_comm = MPI_COMM_NULL;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        rank == base_rank ? MPI_UNDEFINED : 0,
        rank,
        &node_comm
    );

    if (rank == base_rank) {
        run_base_station(&cfg);

    } else {
        int dims[2] = {cfg.rows, cfg.cols};
        int periods[2] = {0, 0};

        MPI_Comm cart_comm = MPI_COMM_NULL;

        MPI_Cart_create(
            node_comm, 2, dims, periods,
            0, &cart_comm
        );

        if (cart_comm == MPI_COMM_NULL) {
            fprintf(stderr, "Cartesian topology failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        run_charging_node(rank, cart_comm, &cfg);

        MPI_Comm_free(&node_comm);
    }

    MPI_Finalize();
    return 0;
}
