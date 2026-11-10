/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Replacement for the engine's network layer (net_main.c and drivers).
 *
 * There is exactly one "connection": the player driven through the public
 * API. dqsim.c asks for it to connect and queues the bytes a real client
 * would send (console commands and movement). The unmodified server code
 * then accepts the connection and parses those messages exactly as it would
 * for a network client. Everything the server sends back is discarded.
 */
/* Same include order as upstream net_main.c. */
#include "q_stdinc.h"
#include "arch_def.h"
#include "net_sys.h"
#include "quakedef.h"
#include "net_defs.h"

#include "dq_internal.h"

#define QUEUE_SLOTS DQ_NET_SLOTS
#define SLOT_BYTES DQ_NET_SLOT_BYTES

sizebuf_t net_message;
double net_time;
int net_activeconnections;
cvar_t hostname = {"hostname", "dqsim", CVAR_NONE};
qboolean tcpipAvailable = false;
qboolean ipxAvailable = false;
char my_tcpip_address[NET_NAMELEN];
char my_ipx_address[NET_NAMELEN];

static qsocket_t the_socket;
static qboolean connect_pending;

static byte queue[QUEUE_SLOTS][SLOT_BYTES];
static int queue_length[QUEUE_SLOTS];
static int queue_head, queue_count;

void dq_net_request_connect(void)
{
    connect_pending = true;
    queue_head = queue_count = 0;
}

/* Snapshot support: everything queued but not yet read by the server. */
void dq_net_get_state(dq_net_state *out)
{
    memcpy(out->queue, queue, sizeof(queue));
    memcpy(out->queue_length, queue_length, sizeof(queue_length));
    out->queue_head = queue_head;
    out->queue_count = queue_count;
    out->connect_pending = connect_pending;
}

void dq_net_set_state(const dq_net_state *in)
{
    memcpy(queue, in->queue, sizeof(queue));
    memcpy(queue_length, in->queue_length, sizeof(queue_length));
    queue_head = in->queue_head;
    queue_count = in->queue_count;
    connect_pending = in->connect_pending;
}

void dq_net_queue(const void *data, int length)
{
    if (length > SLOT_BYTES || queue_count == QUEUE_SLOTS)
        Sys_Error("dq_net_queue: message queue overflow");
    int slot = (queue_head + queue_count) % QUEUE_SLOTS;
    memcpy(queue[slot], data, length);
    queue_length[slot] = length;
    queue_count++;
}

void NET_Init(void)
{
    SZ_Alloc(&net_message, NET_MAXMESSAGE);
    Cvar_RegisterVariable(&hostname);
}

void NET_Shutdown(void)
{
}

void NET_Poll(void)
{
}

struct qsocket_s *NET_CheckNewConnections(void)
{
    if (!connect_pending)
        return NULL;
    connect_pending = false;
    memset(&the_socket, 0, sizeof(the_socket));
    q_strlcpy(the_socket.address, "local", sizeof(the_socket.address));
    return &the_socket;
}

void NET_Close(struct qsocket_s *sock)
{
    if (sock)
        sock->disconnected = true;
    queue_head = queue_count = 0;
}

/* Returns 0 for no message, 1 for a message now in net_message. */
int NET_GetMessage(struct qsocket_s *sock)
{
    (void)sock;
    if (queue_count == 0)
        return 0;
    SZ_Clear(&net_message);
    SZ_Write(&net_message, queue[queue_head], queue_length[queue_head]);
    queue_head = (queue_head + 1) % QUEUE_SLOTS;
    queue_count--;
    return 1;
}

int NET_SendMessage(struct qsocket_s *sock, sizebuf_t *data)
{
    (void)sock;
    (void)data;
    return 1;
}

int NET_SendUnreliableMessage(struct qsocket_s *sock, sizebuf_t *data)
{
    (void)sock;
    (void)data;
    return 1;
}

qboolean NET_CanSendMessage(struct qsocket_s *sock)
{
    (void)sock;
    return true;
}

int NET_SendToAll(sizebuf_t *data, double blocktime)
{
    (void)data;
    (void)blocktime;
    return 0;
}

const char *NET_QSocketGetAddressString(const struct qsocket_s *sock)
{
    return sock->address;
}

double NET_QSocketGetTime(const struct qsocket_s *sock)
{
    (void)sock;
    return 0.0;
}
