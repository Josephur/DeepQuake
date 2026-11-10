/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * libdqsim: headless Quake simulator for DeepQuake.
 *
 * This header is the only interface that code outside engine/ may depend on.
 * The simulation API (create, load map, reset, step, observe) is added in
 * roadmap milestone M1.
 */
#ifndef DQSIM_H
#define DQSIM_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  ifdef DQSIM_BUILDING
#    define DQSIM_API __declspec(dllexport)
#  else
#    define DQSIM_API __declspec(dllimport)
#  endif
#else
#  define DQSIM_API __attribute__((visibility("default")))
#endif

/* Library version as "major.minor.patch". The string is static. */
DQSIM_API const char *dqsim_version(void);

#ifdef __cplusplus
}
#endif

#endif /* DQSIM_H */
