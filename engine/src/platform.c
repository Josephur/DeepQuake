/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * System and console layer for the headless build: the Sys_* and Con_*
 * functions that upstream implements in sys_sdl_unix.c and console.c.
 *
 * Differences from upstream, all deliberate:
 *  - Sys_Error and Sys_Quit do not exit the process. They record a message
 *    and jump back to the public API function that was running.
 *  - There is no wall clock. Sys_DoubleTime returns 0.
 *  - Console output is dropped unless the simulator was started verbose.
 */
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>

#include "quakedef.h"

#include "dq_internal.h"

jmp_buf dq_fatal_jmp;
char dq_error[1024];
int dq_verbose;

qboolean isDedicated = true;
qboolean con_initialized = false;
cvar_t sys_throttle = {"sys_throttle", "0.02", CVAR_ARCHIVE};

/* ---- file handles (same contract as upstream sys_sdl_unix.c) ---- */

#define MAX_HANDLES 32
static FILE *sys_handles[MAX_HANDLES];

static int find_handle(void)
{
    for (int i = 1; i < MAX_HANDLES; i++) {
        if (!sys_handles[i])
            return i;
    }
    Sys_Error("out of handles");
}

int Sys_FileOpenRead(const char *path, int *hndl)
{
    int i = find_handle();
    FILE *f = fopen(path, "rb");
    if (!f) {
        *hndl = -1;
        return -1;
    }
    sys_handles[i] = f;
    *hndl = i;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    return (int)length;
}

int Sys_FileOpenWrite(const char *path)
{
    int i = find_handle();
    FILE *f = fopen(path, "wb");
    if (!f)
        Sys_Error("Error opening %s: %s", path, strerror(errno));
    sys_handles[i] = f;
    return i;
}

void Sys_FileClose(int handle)
{
    fclose(sys_handles[handle]);
    sys_handles[handle] = NULL;
}

void Sys_FileSeek(int handle, int position)
{
    fseek(sys_handles[handle], position, SEEK_SET);
}

int Sys_FileRead(int handle, void *dest, int count)
{
    return (int)fread(dest, 1, count, sys_handles[handle]);
}

int Sys_FileWrite(int handle, const void *data, int count)
{
    return (int)fwrite(data, 1, count, sys_handles[handle]);
}

int Sys_FileType(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return FS_ENT_NONE;
    if (S_ISDIR(st.st_mode))
        return FS_ENT_DIRECTORY;
    if (S_ISREG(st.st_mode))
        return FS_ENT_FILE;
    return FS_ENT_NONE;
}

void Sys_mkdir(const char *path)
{
    if (mkdir(path, 0777) != 0 && errno != EEXIST)
        Sys_Error("Unable to create directory %s: %s", path, strerror(errno));
}

/* ---- errors, time, output ---- */

void Sys_Error(const char *error, ...)
{
    va_list args;
    va_start(args, error);
    vsnprintf(dq_error, sizeof(dq_error), error, args);
    va_end(args);
    if (dq_verbose)
        fprintf(stderr, "dqsim fatal: %s\n", dq_error);
    longjmp(dq_fatal_jmp, 1);
}

void Sys_Quit(void)
{
    snprintf(dq_error, sizeof(dq_error), "the engine executed the 'quit' command");
    longjmp(dq_fatal_jmp, 1);
}

double Sys_DoubleTime(void)
{
    return 0.0;
}

const char *Sys_ConsoleInput(void)
{
    return NULL;
}

void Sys_SendKeyEvents(void)
{
}

static void print_console(const char *fmt, va_list args)
{
    if (dq_verbose)
        vfprintf(stderr, fmt, args);
}

#define DEFINE_PRINT(name, condition)                                          \
    void name(const char *fmt, ...)                                            \
    {                                                                          \
        if (!(condition))                                                      \
            return;                                                            \
        va_list args;                                                          \
        va_start(args, fmt);                                                   \
        print_console(fmt, args);                                              \
        va_end(args);                                                          \
    }

DEFINE_PRINT(Sys_Printf, 1)
DEFINE_PRINT(Con_Printf, 1)
DEFINE_PRINT(Con_SafePrintf, 1)
DEFINE_PRINT(Con_Warning, 1)
DEFINE_PRINT(Con_DPrintf, developer.value)
DEFINE_PRINT(Con_DWarning, developer.value)
DEFINE_PRINT(Con_DPrintf2, developer.value >= 2)

void Con_Init(void)
{
}

void LOG_Init(quakeparms_t *parms)
{
    (void)parms;
}

void LOG_Close(void)
{
}
