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

/* net_fake.c: the single in-process "connection" that carries player input */
void dq_net_request_connect(void);
void dq_net_queue(const void *data, int length);

#endif
