#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_PORTS 6
#define DEFAULT_CYCLES 8
#define DEFAULT_INTERVAL_MS 1000
#define DEFAULT_THRESHOLD 1
#define MAX_NEIGHBOURS 4

/* Message tags */
enum {
    TAG_NEIGHBOUR_REQUEST = 100,
    TAG_NEIGHBOUR_RESPONSE = 101,
    TAG_CONGESTION_REPORT = 200,
    TAG_REDIRECT = 201,
    TAG_TERMINATE = 999
};

typedef struct {
    int node_rank;
    int coord_row;
    int coord_col;
    int local_available;
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

static long long now_epoch_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((long long)ts.tv_sec * 1000LL) + (ts.tv_nsec / 1000000LL);
}

static void format_time(long long epoch_ms, char *buffer, size_t size) {
    time_t seconds = (time_t)(epoch_ms / 1000LL);
    struct tm tm_value;
    localtime_r(&seconds, &tm_value);

    char base[64];
    strftime(base, sizeof(base), "%Y-%m-%d %H:%M:%S", &tm_value);
    snprintf(buffer, size, "%s.%03lld", base, epoch_ms % 1000LL);
}

static void sleep_ms(int milliseconds) {
    if (milliseconds <= 0) return;

    struct timespec req;
    req.tv_sec = milliseconds / 1000;
    req.tv_nsec = (long)(milliseconds % 1000) * 1000000L;

    while (nanosleep(&req, &req) == -1) {
        /* Continue sleeping if interrupted. */
    }
}

static int count_available(const int *port_status, int ports) {
    int total = 0;

    #pragma omp parallel for reduction(+ : total)
    for (int i = 0; i < ports; ++i) {
        total += port_status[i];
    }

    return total;
}

/*
 * Each charging-port state is updated in parallel.
 * 1 = free, 0 = occupied.
 * rand_r() uses a thread-local seed so threads do not race on rand().
 */
static void update_ports(int *port_status, int ports, int world_rank,
                         int cycle, int force_low) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)time(NULL)
                          ^ (unsigned int)(world_rank * 1009)
                          ^ (unsigned int)(cycle * 9176)
                          ^ (unsigned int)(tid * 7919)
                          ^ (unsigned int)now_epoch_ms();

        /* 10% free when force_low=1, otherwise about 55% free. */
        int free_probability = force_low ? 10 : 55;

        #pragma omp for
        for (int i = 0; i < ports; ++i) {
            int value = (int)(rand_r(&seed) % 100U);
            port_status[i] = (value < free_probability) ? 1 : 0;
        }
    }
}

/* Order: top, bottom, left, right. */
static void discover_neighbours(MPI_Comm cart_comm, int neighbours[MAX_NEIGHBOURS]) {
    MPI_Cart_shift(cart_comm, 0, 1, &neighbours[0], &neighbours[1]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbours[2], &neighbours[3]);
}

/*
 * Task-1-only Base Station test harness.
 * This is NOT the full Task 2 Base Station implementation.
 */
static void run_base_station_stub(int world_size, int cycles, int interval_ms) {
    int worker_count = world_size - 1;
    FILE *log = fopen("task1_base_station_stub.log", "w");
    if (log == NULL) {
        fprintf(stderr, "Base Station: could not open log file.\n");
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    fprintf(log, "Task 1 test-harness Base Station\n");
    fprintf(log, "Workers: %d\n\n", worker_count);
    fflush(log);

    double stop_after = (cycles * interval_ms / 1000.0) + 3.0;
    double started = MPI_Wtime();

    while ((MPI_Wtime() - started) < stop_after) {
        int flag = 0;
        MPI_Status status;

        MPI_Iprobe(MPI_ANY_SOURCE, TAG_CONGESTION_REPORT,
                   MPI_COMM_WORLD, &flag, &status);

        if (!flag) {
            sleep_ms(10);
            continue;
        }

        CongestionReport report;
        memset(&report, 0, sizeof(report));

        MPI_Recv(&report, (int)sizeof(report), MPI_BYTE,
                 status.MPI_SOURCE, TAG_CONGESTION_REPORT,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        RedirectReply reply;
        reply.candidate_count = 0;
        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            reply.candidate_ranks[i] = -1;
        }

        /*
         * The Task 1 stub simply returns neighbours reported as available.
         * The real nearest-node search belongs in Task 2.
         */
        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            if (report.neighbour_ranks[i] != MPI_PROC_NULL &&
                report.neighbour_available[i] > 0 &&
                reply.candidate_count < MAX_NEIGHBOURS) {
                reply.candidate_ranks[reply.candidate_count++] =
                    report.neighbour_ranks[i];
            }
        }

        char timestamp[80];
        format_time(report.epoch_ms, timestamp, sizeof(timestamp));

        fprintf(log,
                "[%s] node=%d coord=(%d,%d) local_available=%d "
                "neighbour_messages=%d neighbour_bytes=%d exchange_s=%.9f "
                "candidates=%d\n",
                timestamp,
                report.node_rank,
                report.coord_row,
                report.coord_col,
                report.local_available,
                report.neighbour_messages,
                report.neighbour_bytes,
                report.neighbour_exchange_seconds,
                reply.candidate_count);
        fflush(log);

        MPI_Send(&reply, (int)sizeof(reply), MPI_BYTE,
                 status.MPI_SOURCE, TAG_REDIRECT, MPI_COMM_WORLD);
    }

    for (int rank = 1; rank < world_size; ++rank) {
        int terminate = 1;
        MPI_Send(&terminate, 1, MPI_INT,
                 rank, TAG_TERMINATE, MPI_COMM_WORLD);
    }

    fprintf(log, "Termination sent to all EV charging nodes.\n");
    fclose(log);
}

static void run_charging_node(int world_rank,
                              int worker_rank,
                              MPI_Comm cart_comm,
                              int ports,
                              int cycles,
                              int interval_ms,
                              int threshold,
                              int force_low) {
    int cart_rank = -1;
    int coords[2] = {-1, -1};
    int neighbours[MAX_NEIGHBOURS];

    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    discover_neighbours(cart_comm, neighbours);

    int *port_status = (int *)malloc((size_t)ports * sizeof(int));
    if (port_status == NULL) {
        fprintf(stderr, "Rank %d: could not allocate port array.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 11);
    }
    for (int i = 0; i < ports; ++i) {
        port_status[i] = 1;
    }

    char filename[128];
    snprintf(filename, sizeof(filename), "task1_node_%d.log", world_rank);
    FILE *log = fopen(filename, "w");
    if (log == NULL) {
        fprintf(stderr, "Rank %d: could not open node log.\n", world_rank);
        free(port_status);
        MPI_Abort(MPI_COMM_WORLD, 12);
    }

    fprintf(log,
            "world_rank=%d worker_rank=%d cart_rank=%d coords=(%d,%d)\n",
            world_rank, worker_rank, cart_rank, coords[0], coords[1]);
    fprintf(log,
            "neighbours[top,bottom,left,right]=[%d,%d,%d,%d]\n",
            neighbours[0], neighbours[1], neighbours[2], neighbours[3]);
    fflush(log);

    int termination_received = 0;

    for (int cycle = 0; cycle < cycles && !termination_received; ++cycle) {
        int terminate_flag = 0;
        MPI_Status terminate_status;

        MPI_Iprobe(0, TAG_TERMINATE, MPI_COMM_WORLD,
                   &terminate_flag, &terminate_status);

        if (terminate_flag) {
            int value = 0;
            MPI_Recv(&value, 1, MPI_INT, 0, TAG_TERMINATE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            termination_received = 1;
            break;
        }

        update_ports(port_status, ports, world_rank, cycle, force_low);
        int local_available = count_available(port_status, ports);

        long long timestamp_ms = now_epoch_ms();
        char timestamp[80];
        format_time(timestamp_ms, timestamp, sizeof(timestamp));

        fprintf(log, "[%s] cycle=%d local_available=%d/%d\n",
                timestamp, cycle, local_available, ports);
        fflush(log);

        int congested = (local_available <= threshold);
        int request_payload = congested ? 1 : 0;

        /* Phase 1: exchange neighbour-request flags. */
        int incoming_requests[MAX_NEIGHBOURS] = {0, 0, 0, 0};
        MPI_Request phase1_reqs[MAX_NEIGHBOURS * 2];
        int phase1_count = 0;

        double neighbour_start = MPI_Wtime();

        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            if (neighbours[i] == MPI_PROC_NULL) continue;
            MPI_Irecv(&incoming_requests[i], 1, MPI_INT,
                      neighbours[i], TAG_NEIGHBOUR_REQUEST,
                      cart_comm, &phase1_reqs[phase1_count++]);
        }

        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            if (neighbours[i] == MPI_PROC_NULL) continue;
            MPI_Isend(&request_payload, 1, MPI_INT,
                      neighbours[i], TAG_NEIGHBOUR_REQUEST,
                      cart_comm, &phase1_reqs[phase1_count++]);
        }

        if (phase1_count > 0) {
            MPI_Waitall(phase1_count, phase1_reqs, MPI_STATUSES_IGNORE);
        }

        /* Phase 2: congested nodes receive availability responses. */
        int neighbour_available[MAX_NEIGHBOURS];
        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            neighbour_available[i] = -1;
        }

        MPI_Request phase2_reqs[MAX_NEIGHBOURS * 2];
        int phase2_count = 0;

        if (congested) {
            for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
                if (neighbours[i] == MPI_PROC_NULL) continue;
                MPI_Irecv(&neighbour_available[i], 1, MPI_INT,
                          neighbours[i], TAG_NEIGHBOUR_RESPONSE,
                          cart_comm, &phase2_reqs[phase2_count++]);
            }
        }

        for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
            if (neighbours[i] == MPI_PROC_NULL || incoming_requests[i] == 0) {
                continue;
            }

            MPI_Isend(&local_available, 1, MPI_INT,
                      neighbours[i], TAG_NEIGHBOUR_RESPONSE,
                      cart_comm, &phase2_reqs[phase2_count++]);
        }

        if (phase2_count > 0) {
            MPI_Waitall(phase2_count, phase2_reqs, MPI_STATUSES_IGNORE);
        }

        double neighbour_end = MPI_Wtime();

        if (congested) {
            int any_neighbour_available = 0;

            for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
                if (neighbours[i] != MPI_PROC_NULL &&
                    neighbour_available[i] > threshold) {
                    any_neighbour_available = 1;
                }
            }

            if (any_neighbour_available) {
                fprintf(log,
                        "  Redirect locally: at least one neighbour has available ports.\n");
            } else {
                CongestionReport report;
                memset(&report, 0, sizeof(report));

                report.node_rank = world_rank;
                report.coord_row = coords[0];
                report.coord_col = coords[1];
                report.local_available = local_available;
                report.epoch_ms = now_epoch_ms();
                report.neighbour_exchange_seconds =
                    neighbour_end - neighbour_start;

                for (int i = 0; i < MAX_NEIGHBOURS; ++i) {
                    report.neighbour_ranks[i] = neighbours[i];
                    report.neighbour_available[i] = neighbour_available[i];

                    if (neighbours[i] != MPI_PROC_NULL) {
                        report.valid_neighbour_count++;
                        /* One request int + one response int. */
                        report.neighbour_messages += 2;
                        report.neighbour_bytes += 2 * (int)sizeof(int);
                    }
                }

                MPI_Send(&report, (int)sizeof(report), MPI_BYTE,
                         0, TAG_CONGESTION_REPORT, MPI_COMM_WORLD);

                RedirectReply reply;
                memset(&reply, 0, sizeof(reply));

                MPI_Recv(&reply, (int)sizeof(reply), MPI_BYTE,
                         0, TAG_REDIRECT, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                fprintf(log,
                        "  Congestion report sent to Base Station. "
                        "returned_candidates=%d [",
                        reply.candidate_count);

                for (int i = 0; i < reply.candidate_count; ++i) {
                    if (i > 0) fprintf(log, ",");
                    fprintf(log, "%d", reply.candidate_ranks[i]);
                }
                fprintf(log, "]\n");
            }
            fflush(log);
        }

        sleep_ms(interval_ms);
    }

    /* If the termination message did not arrive early, wait for it now. */
    if (!termination_received) {
        int value = 0;
        MPI_Recv(&value, 1, MPI_INT, 0, TAG_TERMINATE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    fprintf(log, "Graceful shutdown: received termination, closing node.\n");
    fclose(log);
    free(port_status);
}

static void usage(const char *exe) {
    fprintf(stderr,
            "Usage: mpirun -np <rows*cols+1> %s "
            "<rows> <cols> [ports=6] [cycles=8] [interval_ms=1000] "
            "[threshold=1] [force_low=0]\n",
            exe);
    fprintf(stderr,
            "Example: mpirun -np 10 %s 3 3 6 8 500 1 1\n",
            exe);
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = -1;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            fprintf(stderr,
                    "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (argc < 3) {
        if (world_rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int ports = (argc >= 4) ? atoi(argv[3]) : DEFAULT_PORTS;
    int cycles = (argc >= 5) ? atoi(argv[4]) : DEFAULT_CYCLES;
    int interval_ms = (argc >= 6) ? atoi(argv[5]) : DEFAULT_INTERVAL_MS;
    int threshold = (argc >= 7) ? atoi(argv[6]) : DEFAULT_THRESHOLD;
    int force_low = (argc >= 8) ? (atoi(argv[7]) != 0) : 0;

    int expected_world_size = rows * cols + 1;

    if (rows <= 0 || cols <= 0 || ports <= 0 || cycles <= 0 ||
        interval_ms < 0 || threshold < 0 || threshold >= ports ||
        world_size != expected_world_size) {

        if (world_rank == 0) {
            fprintf(stderr,
                    "Invalid configuration. Expected MPI ranks = rows*cols + 1 = %d, "
                    "but got %d. Also require 0 <= threshold < ports.\n",
                    expected_world_size, world_size);
            usage(argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    /*
     * Rank 0 is the Task-1-only Base Station test harness.
     * All other ranks are EV charging nodes.
     */
    MPI_Comm worker_comm = MPI_COMM_NULL;
    int worker_color = (world_rank == 0) ? MPI_UNDEFINED : 1;
    MPI_Comm_split(MPI_COMM_WORLD, worker_color, world_rank, &worker_comm);

    if (world_rank == 0) {
        run_base_station_stub(world_size, cycles, interval_ms);
        MPI_Finalize();
        return 0;
    }

    int worker_rank = -1;
    MPI_Comm_rank(worker_comm, &worker_rank);

    int dims[2] = {rows, cols};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm = MPI_COMM_NULL;

    int cart_err = MPI_Cart_create(worker_comm, 2, dims, periods,
                                   reorder, &cart_comm);

    if (cart_err != MPI_SUCCESS || cart_comm == MPI_COMM_NULL) {
        fprintf(stderr, "Rank %d: MPI_Cart_create failed.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    run_charging_node(world_rank, worker_rank, cart_comm,
                      ports, cycles, interval_ms, threshold, force_low);

    MPI_Comm_free(&cart_comm);
    MPI_Comm_free(&worker_comm);
    MPI_Finalize();
    return 0;
}
