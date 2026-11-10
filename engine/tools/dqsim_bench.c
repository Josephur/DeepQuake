/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Throughput benchmark: simulation ticks per second.
 *
 *   dqsim_bench <gamedata-dir> [map] [ticks] [processes]
 *
 * Starts the given number of processes, each running its own simulator for
 * the given number of ticks with a moving, shooting player, and reports
 * ticks per second per process and in total. Map loading is not timed.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "dqsim/dqsim.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Runs one simulator; writes its elapsed seconds to the pipe. */
static int worker(const char *basedir, const char *map, int ticks, int index, int out_fd)
{
    dqsim_config config = {.basedir = basedir};
    if (dqsim_init(&config) != DQSIM_OK || dqsim_load_map(map, 1, (uint32_t)index) != DQSIM_OK) {
        fprintf(stderr, "worker %d: %s\n", index, dqsim_last_error());
        return 1;
    }
    double start = now();
    for (int tick = 0; tick < ticks; tick++) {
        dqsim_action action = {.forward = 400, .yaw = (float)((tick + index * 37) % 360)};
        if (tick % 5 == 0)
            action.buttons = DQSIM_BUTTON_ATTACK;
        if (dqsim_step(&action) != DQSIM_OK) {
            fprintf(stderr, "worker %d: %s\n", index, dqsim_last_error());
            return 1;
        }
    }
    double elapsed = now() - start;
    if (write(out_fd, &elapsed, sizeof(elapsed)) != (ssize_t)sizeof(elapsed))
        return 1;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gamedata-dir> [map] [ticks] [processes]\n", argv[0]);
        return 2;
    }
    const char *basedir = argv[1];
    const char *map = argc > 2 ? argv[2] : "e1m1";
    int ticks = argc > 3 ? atoi(argv[3]) : 20000;
    int processes = argc > 4 ? atoi(argv[4]) : 1;
    if (ticks < 1 || processes < 1 || processes > 1024) {
        fprintf(stderr, "ticks must be >= 1 and processes 1-1024\n");
        return 2;
    }

    int fds[2];
    if (pipe(fds) != 0) {
        perror("pipe");
        return 1;
    }
    double start = now();
    for (int i = 0; i < processes; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0)
            _exit(worker(basedir, map, ticks, i, fds[1]));
    }
    close(fds[1]);

    int failed = 0;
    for (int i = 0; i < processes; i++) {
        int status;
        wait(&status);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            failed++;
    }
    double wall = now() - start;
    if (failed) {
        fprintf(stderr, "%d of %d workers failed\n", failed, processes);
        return 1;
    }

    double stepping = 0, elapsed;
    while (read(fds[0], &elapsed, sizeof(elapsed)) == (ssize_t)sizeof(elapsed))
        stepping += elapsed;
    double per_process = ticks / (stepping / processes);
    printf("map=%s ticks=%d processes=%d\n", map, ticks, processes);
    printf("ticks/s per process: %.0f\n", per_process);
    printf("ticks/s total:       %.0f\n", per_process * processes);
    printf("simulated-to-real time ratio per process: %.0fx\n",
           per_process * DQSIM_DEFAULT_TICK_SECONDS);
    printf("wall time including start-up and map load: %.2f s\n", wall);
    return 0;
}
