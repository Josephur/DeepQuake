/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Stand-in for <SDL.h> so vendored QuakeSpasm sources build without SDL.
 * Only what the vendored server-side files use: SDL_RWops file reading
 * (common.c, localization loading), implemented over stdio.
 */
#ifndef DQSIM_SHIM_SDL_H
#define DQSIM_SHIM_SDL_H

#include <stdint.h>
#include <stdio.h>

typedef int32_t Sint32;
typedef int64_t Sint64;
typedef FILE SDL_RWops;

#define RW_SEEK_SET SEEK_SET
#define RW_SEEK_END SEEK_END

static inline SDL_RWops *SDL_RWFromFile(const char *path, const char *mode)
{
    return fopen(path, mode);
}

static inline Sint64 SDL_RWseek(SDL_RWops *rw, Sint64 offset, int whence)
{
    if (fseek(rw, (long)offset, whence) != 0)
        return -1;
    return ftell(rw);
}

static inline Sint64 SDL_RWtell(SDL_RWops *rw)
{
    return ftell(rw);
}

static inline size_t SDL_RWread(SDL_RWops *rw, void *buf, size_t size, size_t n)
{
    return fread(buf, size, n, rw);
}

static inline Sint64 SDL_RWsize(SDL_RWops *rw)
{
    long pos = ftell(rw);
    if (pos < 0 || fseek(rw, 0, SEEK_END) != 0)
        return -1;
    long size = ftell(rw);
    fseek(rw, pos, SEEK_SET);
    return size;
}

static inline int SDL_RWclose(SDL_RWops *rw)
{
    return fclose(rw);
}

#endif
