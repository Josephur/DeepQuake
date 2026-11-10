/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Stand-in for QuakeSpasm glquake.h, which is not vendored because it
 * declares the whole OpenGL renderer. This declares only what the vendored
 * server-side files reference; the definitions live in ../client_stubs.c.
 * Struct layouts and constants are copied from upstream glquake.h.
 */
#ifndef DQSIM_SHIM_GLQUAKE_H
#define DQSIM_SHIM_GLQUAKE_H

#define ALIAS_BASE_SIZE_RATIO (1.0 / 11.0)
#define MAX_LBM_HEIGHT 480

typedef struct {
    int packetsize;
    int edicts;
    int visedicts;
    int efrags;
    int tempents;
    int beams;
    int dlights;
} devstats_t;
extern devstats_t dev_stats, dev_peakstats;

typedef struct {
    double packetsize;
    double efrags;
    double beams;
    double varstring;
} overflowtimes_t;
extern overflowtimes_t dev_overflows;
#define CONSOLE_RESPAM_TIME 3

extern cvar_t gl_subdivide_size;
extern int gl_warpimagesize;
extern void (*GL_GenerateMipmap)(GLenum target);

void R_NewGame(void);
void CL_RunParticles(void);
void GLMesh_DeleteVertexBuffers(void);
void GL_SubdivideSurface(msurface_t *fa);
void GL_MakeAliasModelDisplayLists(qmodel_t *m, aliashdr_t *hdr);
void Sky_ClearAll(void);
void Sky_LoadTexture(qmodel_t *m, texture_t *mt);
void Sky_LoadTextureQ64(qmodel_t *m, texture_t *mt);

#endif
