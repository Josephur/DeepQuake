/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Definitions for the client-side symbols that the vendored server-side
 * files reference: renderer, sound, input, menus, textures. In a headless
 * simulator they have nothing to do, so they are empty, with two exceptions
 * noted below. The engine runs in dedicated-server mode, where upstream
 * itself skips nearly all of these.
 */
#include "quakedef.h"

#include "miniz.h"

/* ---- state the server code reads ---- */

client_static_t cls;
client_state_t cl;
/* Host_ServerFrame only runs physics for a single player when this is key_game. */
keydest_t key_dest = key_game;
kbutton_t in_mlook;

cvar_t cl_name = {"_cl_name", "player", CVAR_NONE};
cvar_t cl_color = {"_cl_color", "0", CVAR_NONE};
cvar_t cl_startdemos = {"cl_startdemos", "0", CVAR_NONE};
cvar_t gl_subdivide_size = {"gl_subdivide_size", "128", CVAR_NONE};
cvar_t r_nolerp_list = {"r_nolerp_list", "", CVAR_NONE};
cvar_t r_noshadow_list = {"r_noshadow_list", "", CVAR_NONE};

vec3_t r_origin, vpn, vright, vup;
unsigned int d_8to24table[256];
float scr_centertime_off;
qboolean scr_disabled_for_loading;
int gl_warpimagesize;
void (*GL_GenerateMipmap)(GLenum target);

/*
 * Exception 1: V_CalcRoll is used by the server (sv_user.c) to set the
 * player's roll angle, so it must compute the real value. Copied from
 * upstream view.c, with cl_rollangle and cl_rollspeed at their defaults.
 */
float V_CalcRoll(vec3_t angles, vec3_t velocity)
{
    const float rollangle = 2.0f;
    const float rollspeed = 200.0f;
    vec3_t forward, right, up;

    AngleVectors(angles, forward, right, up);
    float side = DotProduct(velocity, right);
    float sign = side < 0 ? -1 : 1;
    side = fabs(side);

    if (side < rollspeed)
        side = side * rollangle / rollspeed;
    else
        side = rollangle;

    return side * sign;
}

/* Exception 2: upstream returns its argument unchanged when not padding. */
int TexMgr_PadConditional(int s)
{
    return s;
}

/* ---- everything else does nothing ---- */

#define STUB(name) void name(void) {}

STUB(CL_ClearSignons)
STUB(CL_DecayLights)
STUB(CL_Disconnect)
STUB(CL_Disconnect_f)
STUB(CL_Init)
STUB(CL_NextDemo)
STUB(CL_RunParticles)
STUB(CL_SendCmd)
STUB(CL_StopPlayback)
STUB(BGM_Shutdown)
STUB(BGM_Update)
STUB(CDAudio_Shutdown)
STUB(CDAudio_Update)
STUB(Chase_Init)
STUB(D_FlushCaches)
STUB(Draw_Init)
STUB(Draw_NewGame)
STUB(GLMesh_DeleteVertexBuffers)
STUB(History_Shutdown)
STUB(IN_Activate)
STUB(IN_Commands)
STUB(IN_Init)
STUB(IN_Shutdown)
STUB(IN_UpdateInputMode)
STUB(Key_Init)
STUB(Key_UpdateForDest)
STUB(M_Init)
STUB(M_Menu_Quit_f)
STUB(R_Init)
STUB(R_NewGame)
STUB(Sbar_Init)
STUB(SCR_BeginLoadingPlaque)
STUB(SCR_EndLoadingPlaque)
STUB(SCR_Init)
STUB(SCR_UpdateScreen)
STUB(S_Init)
STUB(S_Shutdown)
STUB(Sky_ClearAll)
STUB(TexMgr_Init)
STUB(TexMgr_NewGame)
STUB(V_Init)
STUB(VID_Init)
STUB(VID_Lock)
STUB(VID_Shutdown)
STUB(W_LoadWadFile)

qboolean BGM_Init(void)
{
    return false;
}

int CDAudio_Init(void)
{
    return -1;
}

int CL_ReadFromServer(void)
{
    return 0;
}

void CL_EstablishConnection(const char *host)
{
    (void)host;
}

void Key_WriteBindings(FILE *f)
{
    (void)f;
}

void S_Update(vec3_t origin, vec3_t forward, vec3_t right, vec3_t up)
{
    (void)origin;
    (void)forward;
    (void)right;
    (void)up;
}

void GL_SubdivideSurface(msurface_t *fa)
{
    (void)fa;
}

void GL_MakeAliasModelDisplayLists(qmodel_t *m, aliashdr_t *hdr)
{
    (void)m;
    (void)hdr;
}

void Sky_LoadTexture(qmodel_t *m, texture_t *mt)
{
    (void)m;
    (void)mt;
}

void Sky_LoadTextureQ64(qmodel_t *m, texture_t *mt)
{
    (void)m;
    (void)mt;
}

byte *Image_LoadImage(const char *name, int *width, int *height)
{
    (void)name;
    (void)width;
    (void)height;
    return NULL;
}

void TexMgr_FreeTexturesForOwner(qmodel_t *owner)
{
    (void)owner;
}

/* Upstream also returns NULL here when running as a dedicated server. */
gltexture_t *TexMgr_LoadImage(qmodel_t *owner, const char *name, int width, int height,
                              enum srcformat format, byte *data, const char *source_file,
                              src_offset_t source_offset, unsigned flags)
{
    (void)owner;
    (void)name;
    (void)width;
    (void)height;
    (void)format;
    (void)data;
    (void)source_file;
    (void)source_offset;
    (void)flags;
    return NULL;
}

wad_t *W_LoadWadList(const char *names)
{
    (void)names;
    return NULL;
}

void W_FreeWadList(wad_t *wads)
{
    (void)wads;
}

lumpinfo_t *W_GetLumpinfoList(wad_t *wads, const char *name, wad_t **out_wad)
{
    (void)wads;
    (void)name;
    (void)out_wad;
    return NULL;
}

/* Zip reading is only used for the rerelease's localisation archive. */
mz_bool mz_zip_reader_init(mz_zip_archive *pZip, mz_uint64 size, mz_uint flags)
{
    (void)pZip;
    (void)size;
    (void)flags;
    return MZ_FALSE;
}

mz_bool mz_zip_reader_end(mz_zip_archive *pZip)
{
    (void)pZip;
    return MZ_FALSE;
}

void *mz_zip_reader_extract_file_to_heap(mz_zip_archive *pZip, const char *pFilename,
                                         size_t *pSize, mz_uint flags)
{
    (void)pZip;
    (void)pFilename;
    (void)pSize;
    (void)flags;
    return NULL;
}
