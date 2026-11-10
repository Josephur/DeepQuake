/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <stdio.h>
#include <string.h>

#include "dqsim/dqsim.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <expected-version>\n", argv[0]);
        return 2;
    }
    const char *got = dqsim_version();
    if (strcmp(got, argv[1]) != 0) {
        fprintf(stderr, "dqsim_version() = \"%s\", expected \"%s\"\n", got, argv[1]);
        return 1;
    }
    return 0;
}
