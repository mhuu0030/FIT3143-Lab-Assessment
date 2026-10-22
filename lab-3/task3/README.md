# Task 3 communication analysis

This directory implements the communication-delay experiment required by
Task 3. It benchmarks the message payloads and paths used by Tasks 1 and 2:

- neighbour request;
- neighbour response;
- node status update;
- congestion report;
- Task 1 redirect reply;
- Task 2 redirect count;
- Task 2 redirect list;
- completion notification; and
- termination notification.

## Measurement method

Base-station messages use round-trip time divided by two. This avoids relying
on synchronised clocks across physical CAAS nodes. Each process measures its
own interval with `clock_gettime(CLOCK_MONOTONIC)`, which cannot run backwards
in virtualised local environments. Neighbour
messages use matched Cartesian `MPI_Sendrecv` phases; the phase time is
normalised by the number of real directed neighbour messages for that node.
Warm-up measurements are discarded.

The output records the payload size, number of samples, mean, minimum,
maximum, and standard deviation in microseconds. The CSV files therefore
retain the information needed for the Task 3 tables, graphs, and discussion.

## Full CAAS run

Run from this directory:

```bash
sbatch task3_analysis.job
```

The job uses 32 MPI processes: one base station and up to 31 EV nodes. It
measures 30 two-dimensional grid configurations and 30 line configurations,
with nine message types per configuration. Results are written to:

```text
newtask3output/<job-id>/
  task3_grid_results.csv
  task3_line_results.csv
  task3_run_metadata.txt
  task3-analysis-<job-id>.out
  task3-analysis-<job-id>.err
```

Each CSV must contain 271 lines: one header plus 270 measurements. The job
checks this automatically and fails if either dataset is incomplete.

## Local smoke test only

A smaller local run may be used to check compilation and deadlock safety:

```bash
mkdir -p smoke_output
mpicc -O2 -std=c11 -Wall -Wextra -pedantic \
  task3_communication_analysis.c -lm -o task3_communication_analysis
mpirun -np 5 ./task3_communication_analysis 5 2 smoke_output 1
```

Partial local output is not sufficient for the HD rubric. The final analysis
must use the complete CAAS or local-cluster dataset.
