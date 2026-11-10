/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Shared between the files in engine/src. Not part of the public API. */
#ifndef DQ_INTERNAL_H
#define DQ_INTERNAL_H

#include <setjmp.h>
#include <stdint.h>

/* platform.c */
extern jmp_buf dq_fatal_jmp;   /* target of Sys_Error and Sys_Quit */
extern char dq_error[1024];    /* message for dqsim_last_error() */
extern int dq_verbose;         /* print console output to stderr */

/* rng.c: replaces libc rand() in the vendored engine (see CMakeLists.txt) */
void dq_rand_seed(uint32_t seed);
int dqsim_rand(void);
uint64_t dq_rand_state(void);
void dq_rand_set_state(uint64_t state);

/* net_fake.c: the single in-process "connection" that carries player input */
void dq_net_request_connect(void);
void dq_net_queue(const void *data, int length);

/* The fake network's queued client messages, for snapshots. Plain types so
 * files that do not include quakedef.h can use this. */
#define DQ_NET_SLOTS 8
#define DQ_NET_SLOT_BYTES 256
typedef struct dq_net_state {
    unsigned char queue[DQ_NET_SLOTS][DQ_NET_SLOT_BYTES];
    int queue_length[DQ_NET_SLOTS];
    int queue_head;
    int queue_count;
    int connect_pending;
} dq_net_state;

void dq_net_get_state(dq_net_state *out);
void dq_net_set_state(const dq_net_state *in);

#endif
