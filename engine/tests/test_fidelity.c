/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Fidelity test: libdqsim moves the player exactly like stock QuakeSpasm.
 *
 * Method (rerun this after any engine patch):
 *
 *   1. tools/fetch_fidelity_reference.sh builds the pristine upstream
 *      QuakeSpasm at the commit vendored in third_party/quakespasm, using
 *      its own Makefile, into engine/.fidelity/ (gitignored).
 *   2. This test starts that unmodified binary as a dedicated server on a
 *      loopback UDP port: same map, same skill, host_framerate and
 *      sys_ticrate set to 1/72 s, protocol 999 (int32 coordinates, the
 *      finest wire precision stock Quake offers: 1/16 unit).
 *
 *      The map and the cvars are NOT passed as +commands: QuakeSpasm
 *      rebuilds the command line into a 256-byte buffer (CMDLINE_LENGTH in
 *      common.c) and stuffcmds() parses that truncated copy, so with long
 *      absolute paths a trailing "+map e1m1" is silently dropped and the
 *      dedicated server falls back to "map start" - the test then compares
 *      against the wrong level. Instead the server is given a scratch
 *      basedir (a temp dir with "id1" linked to the user's game data) and a
 *      "test" game directory whose autoexec.cfg sets the cvars and loads
 *      the map, which keeps the command line short and out of that buffer.
 *   3. A minimal NetQuake client, implemented below from upstream's
 *      net_dgrm.c / cl_parse.c message formats, performs the normal
 *      sign-on (prespawn/spawn/begin), then sends one clc_move per server
 *      frame with the inputs from fidelity_inputs.txt - byte for byte the
 *      message dqsim_step() queues - and reads the player's position back
 *      from the entity updates, which the server sends for the player's
 *      own edict every frame.
 *   4. The same inputs are replayed through libdqsim in this process.
 *   5. Per-tick origins are compared: the reference wire value quantises
 *      the server float to 1/16 unit, so agreement within that quantum
 *      (plus a small epsilon) at every tick means both engines stepped the
 *      identical physics. Health and on-ground flag must match exactly.
 *
 * The input script and the route stay away from awake monsters (and send
 * "notarget"), because stock QuakeSpasm uses libc rand() while libdqsim
 * redirects rand() to its seeded generator: player movement itself never
 * consults the RNG.
 *
 * Set DQSIM_FIDELITY_DUMP=<path> to write both per-tick series (dqsim and
 * reference) to a file for debugging a mismatch.
 *
 * Exits with 77 (skipped) when the game data or the reference binary is
 * missing. usage: test_fidelity <gamedata-dir> <reference-binary> <inputs>
 */
#define _DEFAULT_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "dqsim/dqsim.h"

/* ------------------------------------------------------------------ */
/* Protocol constants, copied from third_party/quakespasm/Quake/       */
/* protocol.h and net_defs.h (identical to upstream at the vendored    */
/* commit). The engine's wire format is a fixed ABI.                   */
/* ------------------------------------------------------------------ */
enum {
    SVC_NOP = 1, SVC_DISCONNECT = 2, SVC_UPDATESTAT = 3, SVC_VERSION = 4,
    SVC_SETVIEW = 5, SVC_SOUND = 6, SVC_TIME = 7, SVC_PRINT = 8,
    SVC_STUFFTEXT = 9, SVC_SETANGLE = 10, SVC_SERVERINFO = 11,
    SVC_LIGHTSTYLE = 12, SVC_UPDATENAME = 13, SVC_UPDATEFRAGS = 14,
    SVC_CLIENTDATA = 15, SVC_STOPSOUND = 16, SVC_UPDATECOLORS = 17,
    SVC_PARTICLE = 18, SVC_DAMAGE = 19, SVC_SPAWNSTATIC = 20,
    SVC_SPAWNBASELINE = 22, SVC_TEMP_ENTITY = 23, SVC_SETPAUSE = 24,
    SVC_SIGNONNUM = 25, SVC_CENTERPRINT = 26, SVC_KILLEDMONSTER = 27,
    SVC_FOUNDSECRET = 28, SVC_SPAWNSTATICSOUND = 29, SVC_INTERMISSION = 30,
    SVC_FINALE = 31, SVC_CDTRACK = 32, SVC_SELLSCREEN = 33, SVC_CUTSCENE = 34,
    SVC_SKYBOX = 37, SVC_BF = 40, SVC_FOG = 41, SVC_SPAWNBASELINE2 = 42,
    SVC_SPAWNSTATIC2 = 43, SVC_SPAWNSTATICSOUND2 = 44, SVC_LOCALSOUND = 56,
};
enum { CLC_DISCONNECT = 2, CLC_MOVE = 3, CLC_STRINGCMD = 4 };
enum { PROTOCOL_RMQ = 999 };
/* Header flags (net_defs.h). Length is the low 16 bits, big endian. */
enum {
    NET_HEADERSIZE = 8,
    NETFLAG_LENGTH_MASK = 0x0000ffff, NETFLAG_DATA = 0x00010000,
    NETFLAG_ACK = 0x00020000, NETFLAG_EOM = 0x00080000,
    NETFLAG_UNRELIABLE = 0x00100000, NETFLAG_CTL = 0x80000000,
};
enum { CCREQ_CONNECT = 0x01, CCREP_ACCEPT = 0x81, CCREP_REJECT = 0x82 };
enum { NET_PROTOCOL_VERSION = 3 };
/* Entity update bits (protocol.h). */
enum {
    U_MOREBITS = 1 << 0, U_ORIGIN1 = 1 << 1, U_ORIGIN2 = 1 << 2,
    U_ORIGIN3 = 1 << 3, U_ANGLE2 = 1 << 4, U_STEP = 1 << 5,
    U_FRAME = 1 << 6, U_SIGNAL = 1 << 7, U_ANGLE1 = 1 << 8,
    U_ANGLE3 = 1 << 9, U_MODEL = 1 << 10, U_COLORMAP = 1 << 11,
    U_SKIN = 1 << 12, U_EFFECTS = 1 << 13, U_LONGENTITY = 1 << 14,
    U_EXTEND1 = 1 << 15, U_ALPHA = 1 << 16, U_FRAME2 = 1 << 17,
    U_MODEL2 = 1 << 18, U_LERPFINISH = 1 << 19, U_SCALE = 1 << 20,
    U_EXTEND2 = 1 << 23, U_TRANS = 1 << 15,
};
/* svc_clientdata bits (protocol.h). */
enum {
    SU_VIEWHEIGHT = 1 << 0, SU_IDEALPITCH = 1 << 1, SU_PUNCH1 = 1 << 2,
    SU_PUNCH2 = 1 << 3, SU_PUNCH3 = 1 << 4, SU_VELOCITY1 = 1 << 5,
    SU_VELOCITY2 = 1 << 6, SU_VELOCITY3 = 1 << 7, SU_ITEMS = 1 << 9,
    SU_ONGROUND = 1 << 10, SU_INWATER = 1 << 11, SU_WEAPONFRAME = 1 << 12,
    SU_ARMOR = 1 << 13, SU_WEAPON = 1 << 14, SU_EXTEND1 = 1 << 15,
    SU_WEAPON2 = 1 << 16, SU_ARMOR2 = 1 << 17, SU_AMMO2 = 1 << 18,
    SU_SHELLS2 = 1 << 19, SU_NAILS2 = 1 << 20, SU_ROCKETS2 = 1 << 21,
    SU_CELLS2 = 1 << 22, SU_EXTEND2 = 1 << 23, SU_WEAPONFRAME2 = 1 << 24,
    SU_WEAPONALPHA = 1 << 25,
};
/* Sound message field bits and baseline flags (protocol.h). */
enum { SND_VOLUME = 1, SND_ATTENUATION = 2, SND_LARGEENTITY = 8,
       SND_LARGESOUND = 16 };
enum { B_LARGEMODEL = 1, B_LARGEFRAME = 2, B_ALPHA = 4, B_SCALE = 8 };

/* One input line: the same quantities as dqsim_action. */
typedef struct {
    float forward, side, up, pitch, yaw;
    int buttons, impulse;
} input_t;

/* One recorded tick of player state. */
typedef struct {
    float t, origin[3];
    int on_ground, health;
} row_t;

#define MAX_TICKS 4096
#define MAX_ENTITIES 1024

static input_t inputs[MAX_TICKS];
static int n_inputs;

static row_t ref_rows[MAX_TICKS];
static row_t dq_rows[MAX_TICKS];

/* Wire origin resolution of protocol 999: Q_rint(v * 16) on the wire. */
#define QUANTUM (1.0f / 16.0f)
#define TOLERANCE (QUANTUM + 1e-3f)

#define FAIL(...)                                                              \
    do {                                                                       \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                   \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr, "\n");                                                 \
        exit(1);                                                               \
    } while (0)

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void on_alarm(int sig)
{
    (void)sig;
    fprintf(stderr, "FAIL: fidelity test watchdog timeout\n");
    _exit(1);
}

/* ------------------------------------------------------------------ */
/* Inputs                                                              */
/* ------------------------------------------------------------------ */
static void load_inputs(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        FAIL("cannot open inputs file %s", path);
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (n_inputs == MAX_TICKS)
            FAIL("more than %d input lines", MAX_TICKS);
        input_t *in = &inputs[n_inputs];
        if (sscanf(line, "%f %f %f %f %f %d %d", &in->forward, &in->side,
                   &in->up, &in->pitch, &in->yaw, &in->buttons,
                   &in->impulse) != 7)
            FAIL("bad input line %d: %s", n_inputs + 1, line);
        n_inputs++;
    }
    fclose(f);
    if (n_inputs < 100)
        FAIL("only %d inputs, need at least 100", n_inputs);
}

/* ------------------------------------------------------------------ */
/* Payload reading and writing (little endian, as the MSG_* helpers)   */
/* ------------------------------------------------------------------ */
static const unsigned char *msg_p, *msg_end;

static int read_byte(void)
{
    return msg_p < msg_end ? *msg_p++ : -1;
}

static int read_short(void)
{
    if (msg_end - msg_p < 2)
        return -1;
    int v = (int16_t)(msg_p[0] | (msg_p[1] << 8));
    msg_p += 2;
    return v;
}

static int read_long(void)
{
    if (msg_end - msg_p < 4)
        return -1;
    int v = (int32_t)((uint32_t)msg_p[0] | ((uint32_t)msg_p[1] << 8) |
                      ((uint32_t)msg_p[2] << 16) | ((uint32_t)msg_p[3] << 24));
    msg_p += 4;
    return v;
}

static float read_float(void)
{
    int v = read_long();
    float f;
    memcpy(&f, &v, 4);
    return f;
}

/* PRFL_INT32COORD (protocol 999): long * 1/16. */
static float read_coord(void)
{
    return (float)read_long() * (1.0f / 16.0f);
}

/* PRFL_SHORTANGLE: short * 360/65536. */
static float read_angle16(void)
{
    return (float)((int16_t)read_short()) * (360.0f / 65536.0f);
}

static void read_string(char *out, size_t cap)
{
    size_t n = 0;
    int c;
    while ((c = read_byte()) > 0)
        if (n + 1 < cap)
            out[n++] = (char)c;
    out[n] = '\0';
}

static unsigned char wbuf[2048];
static int wlen;

static void w_reset(void) { wlen = 0; }

static void w_byte(int v) { wbuf[wlen++] = (unsigned char)(v & 255); }

static void w_short(int v)
{
    wbuf[wlen++] = (unsigned char)(v & 255);
    wbuf[wlen++] = (unsigned char)((v >> 8) & 255);
}

static void w_long(int32_t v)
{
    for (int i = 0; i < 4; i++)
        wbuf[wlen++] = (unsigned char)(((uint32_t)v >> (8 * i)) & 255);
}

static void w_float(float f)
{
    int32_t v;
    memcpy(&v, &f, 4);
    w_long(v);
}

static void w_string(const char *s)
{
    while (*s)
        wbuf[wlen++] = (unsigned char)*s++;
    wbuf[wlen++] = 0;
}

/* Q_rint: round half away from zero, as upstream mathlib. */
static int q_rint(float f)
{
    return (int)lroundf(f);
}

static void w_angle16(float f)
{
    w_short(q_rint(f * 65536.0f / 360.0f) & 65535);
}

/* Datagram header: two big endian ints, flags|length then sequence. */
static void put_header(unsigned char *p, uint32_t flags_len, uint32_t seq)
{
    p[0] = (unsigned char)(flags_len >> 24);
    p[1] = (unsigned char)(flags_len >> 16);
    p[2] = (unsigned char)(flags_len >> 8);
    p[3] = (unsigned char)flags_len;
    p[4] = (unsigned char)(seq >> 24);
    p[5] = (unsigned char)(seq >> 16);
    p[6] = (unsigned char)(seq >> 8);
    p[7] = (unsigned char)seq;
}

static uint32_t get_be32(const unsigned char *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

/* ------------------------------------------------------------------ */
/* The NetQuake wire client                                            */
/* ------------------------------------------------------------------ */
static int sock = -1;
static struct sockaddr_in server_addr;
static uint32_t send_sequence;    /* reliable messages we have sent */
static uint32_t receive_sequence; /* reliable messages we expect */
static uint32_t unreliable_send_seq;
static int reliable_in_flight;

static int signon_stage;          /* last svc_signonnum value */
static int player_viewentity = 1;
static int playing;               /* signon 4 reached, frames flowing */

/* Tracked entity state: the player's own edict, from the wire. */
static float baseline_origin[MAX_ENTITIES][3];
static float player_origin[3];
static int player_seen;
/* World model announced in svc_serverinfo: the map the server loaded. */
static char ref_map_model[128];

/* Row being filled while parsing one frame. */
static row_t *cur_row;
static int n_rows;

static void udp_send(const unsigned char *data, int len)
{
    if (sendto(sock, data, len, 0, (struct sockaddr *)&server_addr,
               sizeof(server_addr)) < 0)
        FAIL("sendto: %s", strerror(errno));
}

static void send_reliable(const unsigned char *data, int len)
{
    unsigned char pkt[NET_HEADERSIZE + 1024];
    if (reliable_in_flight)
        FAIL("reliable message already in flight");
    if (len > 1024)
        FAIL("reliable payload too large: %d", len);
    put_header(pkt, (NET_HEADERSIZE + len) | NETFLAG_DATA | NETFLAG_EOM,
               send_sequence++);
    memcpy(pkt + NET_HEADERSIZE, data, len);
    udp_send(pkt, NET_HEADERSIZE + len);
    reliable_in_flight = 1;
}

static void send_stringcmd(const char *cmd)
{
    w_reset();
    w_byte(CLC_STRINGCMD);
    w_string(cmd);
    send_reliable(wbuf, wlen);
}

/* One per-tick clc_move, byte for byte the message dqsim_step() queues
 * (src/dqsim.c): clc_move, float time, 3 * Angle16, 3 * short, buttons,
 * impulse - sent unreliably, like CL_SendMove. */
static void send_move(const input_t *in, float cl_time)
{
    unsigned char pkt[NET_HEADERSIZE + 128];
    w_reset();
    w_byte(CLC_MOVE);
    w_float(cl_time);
    w_angle16(in->pitch);
    w_angle16(in->yaw);
    w_angle16(0.0f);
    w_short((int)in->forward);
    w_short((int)in->side);
    w_short((int)in->up);
    w_byte(in->buttons & 3);
    w_byte(in->impulse & 255);
    put_header(pkt, (NET_HEADERSIZE + wlen) | NETFLAG_UNRELIABLE,
               unreliable_send_seq++);
    memcpy(pkt + NET_HEADERSIZE, wbuf, wlen);
    udp_send(pkt, NET_HEADERSIZE + wlen);
}

static void send_ack(uint32_t seq)
{
    unsigned char pkt[NET_HEADERSIZE];
    put_header(pkt, NET_HEADERSIZE | NETFLAG_ACK, seq);
    udp_send(pkt, NET_HEADERSIZE);
}

/* -- server message parsing ---------------------------------------- */
static void parse_clientdata(row_t *row)
{
    int bits = read_short();
    if (bits & SU_EXTEND1)
        bits |= read_byte() << 15;
    if (bits & SU_EXTEND2)
        bits |= read_byte() << 23;
    if (bits & SU_VIEWHEIGHT)
        read_byte();
    if (bits & SU_IDEALPITCH)
        read_byte();
    for (int i = 0; i < 3; i++) {
        if (bits & (SU_PUNCH1 << i))
            read_byte();
        if (bits & (SU_VELOCITY1 << i))
            read_byte();
    }
    read_long(); /* items, always sent */
    if (bits & SU_WEAPONFRAME)
        read_byte();
    if (bits & SU_ARMOR)
        read_byte();
    if (bits & SU_WEAPON)
        read_byte();
    row->health = read_short();
    read_byte(); /* currentammo */
    read_byte(); /* shells */
    read_byte(); /* nails */
    read_byte(); /* rockets */
    read_byte(); /* cells */
    read_byte(); /* weapon */
    if (bits & SU_WEAPON2)
        read_byte();
    if (bits & SU_ARMOR2)
        read_byte();
    if (bits & SU_AMMO2)
        read_byte();
    if (bits & SU_SHELLS2)
        read_byte();
    if (bits & SU_NAILS2)
        read_byte();
    if (bits & SU_ROCKETS2)
        read_byte();
    if (bits & SU_CELLS2)
        read_byte();
    if (bits & SU_WEAPONFRAME2)
        read_byte();
    if (bits & SU_WEAPONALPHA)
        read_byte();
    row->on_ground = (bits & SU_ONGROUND) != 0;
}

static void parse_sound(void)
{
    int mask = read_byte();
    if (mask & SND_VOLUME)
        read_byte();
    if (mask & SND_ATTENUATION)
        read_byte();
    if (mask & SND_LARGEENTITY) {
        read_short();
        read_byte();
    } else {
        read_short(); /* channel with ent packed in the top bits */
    }
    if (mask & SND_LARGESOUND)
        read_short();
    else
        read_byte();
    for (int i = 0; i < 3; i++)
        read_coord();
}

static void parse_temp_entity(void)
{
    int type = read_byte();
    switch (type) {
    case 5: case 6: case 9: case 13: /* lightning/beam: ent, origin3, ent */
        read_byte();
        for (int i = 0; i < 3; i++)
            read_coord();
        read_byte();
        break;
    case 12: /* TE_EXPLOSION2: origin3, colorStart, colorLength */
        for (int i = 0; i < 3; i++)
            read_coord();
        read_byte();
        read_byte();
        break;
    default: /* the rest: origin3 */
        for (int i = 0; i < 3; i++)
            read_coord();
        break;
    }
}

static void parse_update(int bits)
{
    /*
     * The first entity update completes the sign-on, as upstream's
     * CL_ParseUpdate does; the server never sends svc_signonnum 4.
     */
    if (!playing && signon_stage == 3) {
        signon_stage = 4;
        playing = 1;
    }
    if (bits & U_MOREBITS)
        bits |= read_byte() << 8;
    if (bits & U_EXTEND1)
        bits |= read_byte() << 15;
    if (bits & U_EXTEND2)
        bits |= read_byte() << 23;

    int num = (bits & U_LONGENTITY) ? read_short() : read_byte();
    if (num < 0 || num >= MAX_ENTITIES)
        FAIL("entity number %d out of range", num);
    if (bits & U_MODEL)
        read_byte();
    if (bits & U_FRAME)
        read_byte();
    if (bits & U_COLORMAP)
        read_byte();
    if (bits & U_SKIN)
        read_byte();
    if (bits & U_EFFECTS)
        read_byte();

    float origin[3];
    for (int i = 0; i < 3; i++) {
        origin[i] = (bits & (U_ORIGIN1 << i)) ? read_coord()
                                              : baseline_origin[num][i];
        if (i == 0 && (bits & U_ANGLE1))
            read_angle16();
        else if (i == 1 && (bits & U_ANGLE2))
            read_angle16();
        else if (i == 2 && (bits & U_ANGLE3))
            read_angle16();
    }
    if (bits & U_ALPHA)
        read_byte();
    if (bits & U_SCALE)
        read_byte();
    if (bits & U_FRAME2)
        read_byte();
    if (bits & U_MODEL2)
        read_byte();
    if (bits & U_LERPFINISH)
        read_byte();

    if (num == player_viewentity) {
        memcpy(player_origin, origin, sizeof(origin));
        player_seen = 1;
    }
}

static void parse_baseline(int num, int version)
{
    int bits = version == 2 ? read_byte() : 0;
    if (bits & B_LARGEMODEL)
        read_short();
    else
        read_byte();
    if (bits & B_LARGEFRAME)
        read_short();
    else
        read_byte();
    read_byte(); /* colormap */
    read_byte(); /* skin */
    float origin[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        origin[i] = read_coord();
        read_angle16();
    }
    if (bits & B_ALPHA)
        read_byte();
    if (bits & B_SCALE)
        read_byte();
    if (num >= 0) { /* svc_spawnstatic carries no entity number */
        if (num >= MAX_ENTITIES)
            FAIL("baseline entity %d out of range", num);
        memcpy(baseline_origin[num], origin, sizeof(origin));
    }
}

static int server_protocol;

/* Parse one payload; acts on the signon sequence. */
static void parse_message(const unsigned char *data, int len)
{
    msg_p = data;
    msg_end = data + len;
    char str[2048];
    int cmd;
    while ((cmd = read_byte()) != -1) {
        if (cmd & U_SIGNAL) {
            parse_update(cmd & 127);
            continue;
        }
        switch (cmd) {
        case SVC_NOP:
        case SVC_KILLEDMONSTER:
        case SVC_FOUNDSECRET:
        case SVC_INTERMISSION:
        case SVC_SELLSCREEN:
        case SVC_BF:
            break;
        case SVC_DISCONNECT:
            FAIL("server disconnected");
        case SVC_UPDATESTAT:
            read_byte();
            read_long();
            break;
        case SVC_VERSION:
            server_protocol = read_long();
            if (server_protocol != PROTOCOL_RMQ)
                FAIL("server protocol %d, expected %d", server_protocol,
                     PROTOCOL_RMQ);
            break;
        case SVC_SETVIEW:
            player_viewentity = read_short();
            break;
        case SVC_SOUND:
            parse_sound();
            break;
        case SVC_TIME:
            if (cur_row)
                cur_row->t = read_float();
            else
                read_float();
            break;
        case SVC_PRINT:
        case SVC_CENTERPRINT:
        case SVC_STUFFTEXT:
        case SVC_SKYBOX:
            read_string(str, sizeof(str));
            break;
        case SVC_SETANGLE:
            for (int i = 0; i < 3; i++)
                read_angle16();
            break;
        case SVC_SERVERINFO: {
            int proto = read_long();
            if (proto == PROTOCOL_RMQ)
                read_long(); /* protocolflags */
            server_protocol = proto;
            read_byte();     /* maxclients */
            read_byte();     /* gametype */
            read_string(str, sizeof(str));
            /* First precache model is the world, i.e. the map: remember it
             * so main() can check the server ran the level we asked for. */
            for (int model = 0;; model++) {
                read_string(str, sizeof(str));
                if (!str[0])
                    break;
                if (model == 0)
                    snprintf(ref_map_model, sizeof(ref_map_model), "%.100s", str);
            }
            while (1) {
                read_string(str, sizeof(str));
                if (!str[0])
                    break;
            }
            break;
        }
        case SVC_LIGHTSTYLE:
            read_byte();
            read_string(str, sizeof(str));
            break;
        case SVC_UPDATENAME:
            read_byte();
            read_string(str, sizeof(str));
            break;
        case SVC_UPDATEFRAGS:
            read_byte();
            read_short();
            break;
        case SVC_CLIENTDATA:
            parse_clientdata(cur_row ? cur_row : &(row_t){0});
            break;
        case SVC_STOPSOUND:
            read_short();
            break;
        case SVC_UPDATECOLORS:
        case SVC_CDTRACK:
            read_byte();
            read_byte();
            break;
        case SVC_SETPAUSE:
            read_byte();
            break;
        case SVC_PARTICLE:
            for (int i = 0; i < 3; i++)
                read_coord();
            read_byte();
            read_byte();
            read_byte();
            read_byte();
            read_byte();
            break;
        case SVC_DAMAGE:
            read_byte();
            read_byte();
            for (int i = 0; i < 3; i++)
                read_coord();
            break;
        case SVC_SPAWNSTATIC:
            parse_baseline(-1, 1);
            break;
        case SVC_SPAWNSTATIC2:
            parse_baseline(-1, 2);
            break;
        case SVC_SPAWNBASELINE:
            parse_baseline(read_short(), 1); /* entnum is a short (sv_main.c) */
            break;
        case SVC_SPAWNBASELINE2:
            parse_baseline(read_short(), 2);
            break;
        case SVC_TEMP_ENTITY:
            parse_temp_entity();
            break;
        case SVC_SIGNONNUM: {
            int stage = read_byte();
            if (stage <= signon_stage)
                FAIL("signon %d after %d", stage, signon_stage);
            signon_stage = stage;
            if (stage == 1)
                send_stringcmd("prespawn");
            else if (stage == 2)
                send_stringcmd("spawn");
            else if (stage == 3)
                send_stringcmd("begin");
            else if (stage == 4)
                playing = 1;
            break;
        }
        case SVC_SPAWNSTATICSOUND:
            for (int i = 0; i < 3; i++)
                read_coord();
            read_byte();
            read_byte();
            read_byte();
            break;
        case SVC_SPAWNSTATICSOUND2:
            for (int i = 0; i < 3; i++)
                read_coord();
            read_short();
            read_byte();
            read_byte();
            break;
        case SVC_FINALE:
        case SVC_CUTSCENE:
            read_string(str, sizeof(str));
            break;
        case SVC_FOG:
            read_byte();
            read_byte();
            read_byte();
            read_byte();
            read_float();
            break;
        case SVC_LOCALSOUND: {
            int mask = read_byte();
            if (mask & SND_LARGESOUND)
                read_short();
            else
                read_byte();
            break;
        }
        default:
            FAIL("unhandled server message %d at offset %td", cmd,
                 msg_p - data);
        }
    }
    if (msg_p != msg_end)
        FAIL("parse error: %td of %d bytes consumed", msg_p - data, len);
}

/* ------------------------------------------------------------------ */
/* The reference run                                                   */
/* ------------------------------------------------------------------ */
static int udp_recv(unsigned char *buf, int cap, double timeout_s)
{
    struct timeval tv = {.tv_sec = (long)timeout_s,
                         .tv_usec = (long)((timeout_s - (long)timeout_s) *
                                           1e6)};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock, &fds);
    int ready = select(sock + 1, &fds, NULL, NULL, &tv);
    if (ready < 0)
        FAIL("select: %s", strerror(errno));
    if (ready == 0)
        return 0;
    struct sockaddr_in from;
    socklen_t fromlen = sizeof(from);
    ssize_t n = recvfrom(sock, buf, cap, 0, (struct sockaddr *)&from, &fromlen);
    if (n < 0)
        FAIL("recvfrom: %s", strerror(errno));
    if (from.sin_addr.s_addr != server_addr.sin_addr.s_addr ||
        from.sin_port != server_addr.sin_port)
        return 0; /* not from our connection */
    return (int)n;
}

static void run_reference_client(const char *server_ip, int server_port,
                                 double deadline)
{
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
        FAIL("socket: %s", strerror(errno));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons((uint16_t)server_port);
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) != 1)
        FAIL("bad server address %s", server_ip);

    /*
     * Connection request: [CTL|len][CCREQ_CONNECT]["QUAKE"][version].
     * Control datagrams have no sequence word: a 4-byte header only,
     * and the length counts the whole datagram including that header
     * (upstream _Datagram_Connect / _Datagram_CheckNewConnections).
     */
    w_reset();
    w_byte(CCREQ_CONNECT);
    w_string("QUAKE");
    w_byte(NET_PROTOCOL_VERSION);
    static unsigned char pkt[65536 + 16]; /* signon messages can be ~64 KiB */
    put_header(pkt, NETFLAG_CTL | (4 + wlen), 0);
    memcpy(pkt + 4, wbuf, wlen);

    int accepted = 0;
    while (!accepted) {
        if (now_seconds() > deadline)
            FAIL("timeout connecting to the reference server");
        udp_send(pkt, 4 + wlen);
        for (double start = now_seconds(); now_seconds() - start < 1.0 &&
             !accepted;) {
            int n = udp_recv(pkt, sizeof(pkt), 0.25);
            if (n < (int)NET_HEADERSIZE)
                continue;
            uint32_t hdr = get_be32(pkt);
            if (!(hdr & NETFLAG_CTL))
                continue;
            msg_p = pkt + 4; /* control replies: 4-byte header, no sequence */
            msg_end = pkt + n;
            int reply = read_byte();
            if (reply == CCREP_ACCEPT) {
                /* MSG_WriteLong payload: little endian, host order here. */
                int port = read_long();
                server_addr.sin_port = htons((uint16_t)port);
                accepted = 1;
            } else if (reply == CCREP_REJECT) {
                char reason[256];
                read_string(reason, sizeof(reason));
                FAIL("server rejected the connection: %s", reason);
            }
        }
    }

    /*
     * Play loop. Frame datagrams (unreliable, one per server frame, each
     * starting with svc_time) arrive forever; on each, the state of the
     * previous frame's entities is complete, so: parse the frame into a
     * scratch row, commit it as row i if it reflects input i, then send
     * input i + 1. Frame 0 (right after "begin") only triggers the first
     * input, so row i is the state after input i, like the dqsim replay.
     */
    int sent_moves = -1;
    unsigned char frag[65536];
    int frag_len = 0;

    while (1) {
        if (now_seconds() > deadline)
            FAIL("timeout at %d/%d rows (signon stage %d, %s)", n_rows,
                 n_inputs, signon_stage, playing ? "playing" : "signon");
        int n = udp_recv(pkt, sizeof(pkt), 5.0);
        if (n == 0)
            FAIL("no datagram for 5 s (row %d/%d, signon %d)", n_rows,
                 n_inputs, signon_stage);
        if (n < (int)NET_HEADERSIZE)
            continue;
        uint32_t hdr = get_be32(pkt);
        uint32_t flags = hdr & ~NETFLAG_LENGTH_MASK;
        uint32_t length = hdr & NETFLAG_LENGTH_MASK;
        uint32_t seq = get_be32(pkt + 4);
        if (length != (uint32_t)n)
            continue; /* truncated or forged */

        if (flags & NETFLAG_ACK) {
            if (reliable_in_flight && seq + 1 == send_sequence)
                reliable_in_flight = 0;
            continue; /* stale acks are harmless */
        }
        if (flags & NETFLAG_DATA) {
            send_ack(seq);
            if (seq != receive_sequence)
                continue; /* duplicate */
            receive_sequence++;
            int plen = n - NET_HEADERSIZE;
            if (flags & NETFLAG_EOM) {
                if (frag_len) {
                    if (frag_len + plen > (int)sizeof(frag))
                        FAIL("reliable message overflow");
                    memcpy(frag + frag_len, pkt + NET_HEADERSIZE, plen);
                    parse_message(frag, frag_len + plen);
                    frag_len = 0;
                } else {
                    parse_message(pkt + NET_HEADERSIZE, plen);
                }
            } else {
                if (frag_len + plen > (int)sizeof(frag))
                    FAIL("reliable message overflow");
                memcpy(frag + frag_len, pkt + NET_HEADERSIZE, plen);
                frag_len += plen;
            }
            continue;
        }
        if (!(flags & NETFLAG_UNRELIABLE))
            continue;
        if (!playing) {
            /* A frame before "begin" completed; should not happen. */
            parse_message(pkt + NET_HEADERSIZE, n - NET_HEADERSIZE);
            continue;
        }

        row_t scratch = {0};
        cur_row = &scratch;
        parse_message(pkt + NET_HEADERSIZE, n - NET_HEADERSIZE);
        cur_row = NULL;
        scratch.origin[0] = player_origin[0];
        scratch.origin[1] = player_origin[1];
        scratch.origin[2] = player_origin[2];
        if (!player_seen)
            FAIL("no entity update for the player in frame %d", n_rows);

        if (sent_moves >= 0) {
            if (n_rows >= n_inputs)
                FAIL("more frames than inputs");
            ref_rows[n_rows++] = scratch;
        }
        if (sent_moves + 1 < n_inputs) {
            sent_moves++;
            if (sent_moves == 0)
                send_stringcmd("notarget");
            send_move(&inputs[sent_moves], scratch.t);
        } else {
            /* Every input applied and recorded; disconnect and finish. */
            unsigned char bye[NET_HEADERSIZE + 1];
            put_header(bye, (NET_HEADERSIZE + 1) | NETFLAG_DATA | NETFLAG_EOM,
                       send_sequence++);
            bye[NET_HEADERSIZE] = CLC_DISCONNECT;
            udp_send(bye, NET_HEADERSIZE + 1);
            close(sock);
            if (n_rows != n_inputs)
                FAIL("finished with %d rows for %d inputs", n_rows,
                     n_inputs);
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* The libdqsim replay                                                 */
/* ------------------------------------------------------------------ */
static void run_dqsim(const char *gamedata)
{
    dqsim_config config = {.basedir = gamedata};
    if (dqsim_init(&config) != DQSIM_OK)
        FAIL("dqsim_init: %s", dqsim_last_error());
    if (dqsim_load_map("e1m1", 1, 1) != DQSIM_OK)
        FAIL("dqsim_load_map: %s", dqsim_last_error());
    if (dqsim_client_command("notarget") != DQSIM_OK)
        FAIL("dqsim_client_command(notarget): %s", dqsim_last_error());

    /*
     * Let the player settle before the script starts. The reference server
     * runs for a second or two of simulated time while the client signs on,
     * so by the time its first move arrives the player has fallen the last
     * fraction of a unit onto the floor. libdqsim starts stepping three
     * frames after spawn, with the player still in the air; without this the
     * first recorded ticks would compare a falling player with a standing
     * one. Both runs then start the script from the same standing state.
     */
    for (int i = 0; i < 16; i++) {
        dqsim_player_state player;
        if (dqsim_get_player(&player) != DQSIM_OK)
            FAIL("dqsim_get_player while settling: %s", dqsim_last_error());
        if (player.on_ground)
            break;
        dqsim_action idle = {0};
        if (dqsim_step(&idle) != DQSIM_OK)
            FAIL("dqsim_step while settling: %s", dqsim_last_error());
    }

    for (int i = 0; i < n_inputs; i++) {
        dqsim_action action = {.forward = inputs[i].forward,
                               .side = inputs[i].side,
                               .up = inputs[i].up,
                               .pitch = inputs[i].pitch,
                               .yaw = inputs[i].yaw,
                               .buttons = inputs[i].buttons,
                               .impulse = inputs[i].impulse};
        if (dqsim_step(&action) != DQSIM_OK)
            FAIL("dqsim_step %d: %s", i, dqsim_last_error());
        dqsim_player_state player;
        if (dqsim_get_player(&player) != DQSIM_OK)
            FAIL("dqsim_get_player %d: %s", i, dqsim_last_error());
        dq_rows[i].t = (float)dqsim_time();
        dq_rows[i].origin[0] = player.origin[0];
        dq_rows[i].origin[1] = player.origin[1];
        dq_rows[i].origin[2] = player.origin[2];
        dq_rows[i].on_ground = player.on_ground;
        dq_rows[i].health = (int)player.health;
    }
    dqsim_shutdown();
}

/* ------------------------------------------------------------------ */
/* Comparison                                                          */
/* ------------------------------------------------------------------ */
static int compare(void)
{
    float max_dev[3] = {0, 0, 0};
    int first_bad = -1, bad = 0, ground_mismatch = 0, health_mismatch = 0;

    const char *dump_path = getenv("DQSIM_FIDELITY_DUMP");
    if (dump_path && dump_path[0]) {
        FILE *f = fopen(dump_path, "w");
        if (!f)
            FAIL("cannot write %s", dump_path);
        fprintf(f, "# tick t_dq dq_x dq_y dq_z dq_on dq_health "
                   "t_ref ref_x ref_y ref_z ref_on ref_health\n");
        for (int i = 0; i < n_inputs; i++)
            fprintf(f, "%d %.6f %.6f %.6f %.6f %d %d "
                       "%.6f %.6f %.6f %.6f %d %d\n",
                    i, dq_rows[i].t, dq_rows[i].origin[0], dq_rows[i].origin[1],
                    dq_rows[i].origin[2], dq_rows[i].on_ground,
                    dq_rows[i].health, ref_rows[i].t, ref_rows[i].origin[0],
                    ref_rows[i].origin[1], ref_rows[i].origin[2],
                    ref_rows[i].on_ground, ref_rows[i].health);
        fclose(f);
    }

    for (int i = 0; i < n_inputs; i++) {
        int ok = 1;
        for (int k = 0; k < 3; k++) {
            float d = fabsf(dq_rows[i].origin[k] - ref_rows[i].origin[k]);
            if (d > max_dev[k])
                max_dev[k] = d;
            if (d > TOLERANCE)
                ok = 0;
        }
        if (!ok)
            bad++;
        if (dq_rows[i].on_ground != ref_rows[i].on_ground)
            ground_mismatch++;
        if (dq_rows[i].health != ref_rows[i].health)
            health_mismatch++;
        if ((!ok || dq_rows[i].health != ref_rows[i].health) && first_bad < 0)
            first_bad = i;
    }

    printf("fidelity: %d ticks, max |dqsim - reference| per axis "
           "(%.4f, %.4f, %.4f), tolerance %.4f\n",
           n_inputs, max_dev[0], max_dev[1], max_dev[2], TOLERANCE);
    if (bad || ground_mismatch || health_mismatch) {
        printf("FAIL: %d ticks beyond tolerance (first at tick %d), "
               "%d on-ground mismatches, %d health mismatches\n",
               bad, first_bad, ground_mismatch, health_mismatch);
        int shown = 0;
        for (int i = 0; i < n_inputs && shown < 10; i++) {
            float d0 = fabsf(dq_rows[i].origin[0] - ref_rows[i].origin[0]);
            float d1 = fabsf(dq_rows[i].origin[1] - ref_rows[i].origin[1]);
            float d2 = fabsf(dq_rows[i].origin[2] - ref_rows[i].origin[2]);
            if (d0 > TOLERANCE || d1 > TOLERANCE || d2 > TOLERANCE ||
                dq_rows[i].health != ref_rows[i].health) {
                printf("  tick %4d: dqsim (%8.2f %8.2f %8.2f) h=%3d on=%d "
                       "ref (%8.2f %8.2f %8.2f) h=%3d on=%d\n",
                       i, dq_rows[i].origin[0], dq_rows[i].origin[1],
                       dq_rows[i].origin[2], dq_rows[i].health,
                       dq_rows[i].on_ground, ref_rows[i].origin[0],
                       ref_rows[i].origin[1], ref_rows[i].origin[2],
                       ref_rows[i].health, ref_rows[i].on_ground);
                shown++;
            }
        }
        return 1;
    }
    printf("PASS: player positions agree with stock QuakeSpasm at every "
           "tick\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Reference server process                                            */
/* ------------------------------------------------------------------ */
static pid_t server_pid;
static char server_log[128];

static void kill_server(void)
{
    if (server_pid > 0) {
        kill(server_pid, SIGTERM);
        int status;
        struct timespec pause = {.tv_nsec = 100000000};
        for (int i = 0; i < 50; i++) {
            if (waitpid(server_pid, &status, WNOHANG) == server_pid)
                return;
            nanosleep(&pause, NULL);
        }
        kill(server_pid, SIGKILL);
        waitpid(server_pid, &status, 0);
    }
}

/* 1/72 s as a decimal that parses back to the same double the simulator
 * uses (DQSIM_DEFAULT_TICK_SECONDS), for host_framerate and sys_ticrate. */
#define TICK_STRING "0.013888888888888889"

static int find_free_port(void)
{
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0)
        FAIL("socket: %s", strerror(errno));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        FAIL("bind: %s", strerror(errno));
    socklen_t len = sizeof(addr);
    getsockname(s, (struct sockaddr *)&addr, &len);
    int port = ntohs(addr.sin_port);
    close(s);
    return port;
}

/*
 * Scratch basedir for the reference server, set up before the fork:
 *
 *   <tmp>/dqsim_ref_XXXXXX/id1   -> <gamedata>/id1   (the user's paks)
 *   <tmp>/dqsim_ref_XXXXXX/test/autoexec.cfg         (cvars and "map e1m1")
 *
 * "-basedir <tmp> -game test" then keeps the command line short, and the
 * setup travels in the cfg file instead of in "+" arguments that
 * QuakeSpasm's 256-byte command-line buffer can truncate (see the file
 * comment at the top).
 */
#define REF_PATH_MAX 4096
static char ref_basedir[REF_PATH_MAX];
static char ref_game_dir[REF_PATH_MAX + 8];
static char ref_autoexec[REF_PATH_MAX + 24];
static char ref_id1_link[REF_PATH_MAX + 8];

static void cleanup_reference_dir(void)
{
    if (!ref_basedir[0])
        return;
    unlink(ref_autoexec);
    rmdir(ref_game_dir);
    unlink(ref_id1_link);
    rmdir(ref_basedir);
    ref_basedir[0] = 0;
}

static void setup_reference_dir(const char *gamedata)
{
    char data[REF_PATH_MAX];
    if (!realpath(gamedata, data))
        FAIL("realpath(%s): %s", gamedata, strerror(errno));

    snprintf(ref_basedir, sizeof(ref_basedir), "%s/dqsim_ref_XXXXXX",
             P_tmpdir);
    if (!mkdtemp(ref_basedir))
        FAIL("mkdtemp: %s", strerror(errno));
    snprintf(ref_game_dir, sizeof(ref_game_dir), "%s/test", ref_basedir);
    snprintf(ref_autoexec, sizeof(ref_autoexec), "%s/autoexec.cfg",
             ref_game_dir);
    snprintf(ref_id1_link, sizeof(ref_id1_link), "%s/id1", ref_basedir);

    char data_id1[REF_PATH_MAX + 8];
    snprintf(data_id1, sizeof(data_id1), "%s/id1", data);
    if (symlink(data_id1, ref_id1_link) != 0)
        FAIL("symlink(%s): %s", ref_id1_link, strerror(errno));
    if (mkdir(ref_game_dir, 0700) != 0)
        FAIL("mkdir(%s): %s", ref_game_dir, strerror(errno));

    FILE *cfg = fopen(ref_autoexec, "w");
    if (!cfg)
        FAIL("cannot create %s", ref_autoexec);
    fprintf(cfg, "skill 1\n"
                 "host_framerate " TICK_STRING "\n"
                 "sys_ticrate " TICK_STRING "\n"
                 "map e1m1\n");
    fclose(cfg);
}

static void spawn_server(const char *binary, const char *gamedata, int port)
{
    snprintf(server_log, sizeof(server_log), "%s/dqsim_fidelity_%d.log",
             P_tmpdir, (int)getpid());
    FILE *log = fopen(server_log, "w");
    if (!log)
        FAIL("cannot create %s", server_log);
    setup_reference_dir(gamedata);
    server_pid = fork();
    if (server_pid < 0)
        FAIL("fork: %s", strerror(errno));
    if (server_pid == 0) {
        dup2(fileno(log), STDOUT_FILENO);
        dup2(fileno(log), STDERR_FILENO);
        char port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", port);
        /* stdbuf keeps the log line-buffered across exec. */
        execl("/usr/bin/stdbuf", "stdbuf", "-oL", binary, "quakespasm",
              "-dedicated", "1", "-basedir", ref_basedir, "-game", "test",
              "-port", port_str, "-protocol", "999", (char *)NULL);
        _exit(127);
    }
    fclose(log);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <gamedata-dir> <reference-binary> "
                        "<inputs-file>\n",
                argv[0]);
        return 2;
    }
    const char *gamedata = argv[1];
    const char *reference = argv[2];

    char pak[4096];
    snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", gamedata);
    FILE *f = fopen(pak, "rb");
    if (!f) {
        printf("SKIP: no game data at %s\n", pak);
        return 77;
    }
    fclose(f);
    if (access(reference, X_OK) != 0) {
        printf("SKIP: no reference binary at %s\n"
               "      (build it with tools/fetch_fidelity_reference.sh)\n",
               reference);
        return 77;
    }

    load_inputs(argv[3]);
    signal(SIGALRM, on_alarm);
    alarm(300); /* watchdog for the whole comparison */

    int port = find_free_port();
    atexit(cleanup_reference_dir); /* runs after kill_server below */
    atexit(kill_server);
    spawn_server(reference, gamedata, port);
    double deadline = now_seconds() + 120;
    run_reference_client("127.0.0.1", port, deadline);
    kill_server();

    if (strcmp(ref_map_model, "maps/e1m1.bsp") != 0)
        FAIL("reference server loaded '%s', expected maps/e1m1.bsp: it did "
             "not run the map the test asked for",
             ref_map_model[0] ? ref_map_model : "(none)");

    run_dqsim(gamedata);
    int failed = compare();
    if (failed)
        fprintf(stderr, "reference server log: %s\n", server_log);
    else
        unlink(server_log);
    return failed;
}
