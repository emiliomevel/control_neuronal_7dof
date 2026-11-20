/* nn_proto.c -- protocolo PC <-> micro.
 *
 *  | 0xA5 | 0x5A | cmd | seq | len u16 LE | payload | crc16 u16 LE |    CRC-16/CCITT-FALSE sobre cmd..payload
 *  Respuesta: cmd|0x80, mismo seq, payload[0] = status.
 *  Parada y espera: una trama a la vez -> sin desbordes y sin guardar el dataset en el micro.
 *  Un seq repetido (reintento de la PC) reenvia la respuesta anterior SIN volver a ejecutar el comando. */
#include "nn_proto.h"
#include "nn_mlp.h"
#include "bsp.h"
#include <string.h>

enum { CMD_PING = 0x01, CMD_INFO = 0x02, CMD_INIT = 0x03, CMD_SET_LR = 0x04,
       CMD_TRAIN = 0x10, CMD_EVAL = 0x11, CMD_GET_W = 0x20, CMD_SET_W = 0x21, CMD_INFER = 0x30 };
enum { ST_OK = 0, ST_BAD_CMD = 1, ST_BAD_LEN = 2, ST_NOT_AVAILABLE = 3, ST_BAD_ARG = 4 };

#define SOF1 0xA5u
#define SOF2 0x5Au
#define GETW_MAX_CHUNK 256u

static uint16_t crc_tab[256];
static uint8_t  rx_payload[NN_PROTO_MAX_PAYLOAD] __attribute__((aligned(4)));
static uint8_t  tx_frame[NN_PROTO_MAX_PAYLOAD + 8u] __attribute__((aligned(4)));
static uint32_t tx_len;
static uint8_t  last_seq;
static uint8_t  last_valid;
#if NN_TRAIN_ON_BOARD
static float    g_lr = NN_LR_DEFAULT;
#endif

/* ---------------------------------------------------------------- utilidades */
static uint16_t crc16(uint16_t crc, const uint8_t *d, uint32_t n)
{
    while (n--) crc = (uint16_t)((crc << 8) ^ crc_tab[(uint8_t)((crc >> 8) ^ *d++)]);
    return crc;
}

void nn_proto_init(void)
{
    uint32_t i, b;
    for (i = 0; i < 256u; i++) {
        uint16_t c = (uint16_t)(i << 8);
        for (b = 0; b < 8u; b++) c = (c & 0x8000u) ? (uint16_t)((c << 1) ^ 0x1021u) : (uint16_t)(c << 1);
        crc_tab[i] = c;
    }
    last_valid = 0;
}

static void put_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put_f32(uint8_t *p, float v)    { memcpy(p, &v, 4); }
static uint16_t get_u16(const uint8_t *p)   { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t get_u32(const uint8_t *p)   { uint32_t v; memcpy(&v, p, 4); return v; }

static void send_response(uint8_t cmd, uint8_t seq, uint16_t plen)
{
    uint16_t crc;
    tx_frame[0] = SOF1;
    tx_frame[1] = SOF2;
    tx_frame[2] = (uint8_t)(cmd | 0x80u);
    tx_frame[3] = seq;
    put_u16(tx_frame + 4, plen);
    crc = crc16(0xFFFFu, tx_frame + 2, 4u + plen);
    put_u16(tx_frame + 6u + plen, crc);
    tx_len = 8u + plen;
    bsp_uart_write(tx_frame, tx_len);
}

static uint16_t reply_status(uint8_t st) { tx_frame[6] = st; return 1u; }

/* ---------------------------------------------------------------- manejadores */
static uint32_t bench_infer_cycles(void)
{
    static float q[NN_N_JOINTS], dq[NN_N_JOINTS], qr[NN_N_JOINTS], dqr[NN_N_JOINTS], u[NN_N_JOINTS];
    uint32_t j, r, t0;
    for (j = 0; j < NN_N_JOINTS; j++) { q[j] = 0.0f; dq[j] = 0.1f; qr[j] = 0.3f; dqr[j] = 0.1f; }
    t0 = bsp_cycles();
    for (r = 0; r < 32u; r++) nn_infer(q, dq, qr, dqr, u);
    return (bsp_cycles() - t0) / 32u;
}

static uint16_t do_info(void)
{
    uint8_t *p = tx_frame + 6;
    uint8_t flags = 0;
#if NN_TRAIN_ON_BOARD
    flags |= 1u;
#endif
#if NN_SHARED
    flags |= 2u;
#endif
#if NN_IS_RESIDUAL
    flags |= 4u;
#endif
#if NN_USE_ADAM
    flags |= 8u;
#endif
if (nn_weights_valid()) flags |= 16u;
    p[0] = ST_OK;
    p[1] = NN_PROTO_VERSION; p[2] = flags; p[3] = NN_N_JOINTS; p[4] = 0;
    put_u16(p + 5, NN_IN);  put_u16(p + 7, NN_H1);  put_u16(p + 9, NN_H2);  put_u16(p + 11, NN_OUT);
    put_u32(p + 13, NN_N_PARAMS);
    put_u16(p + 17, (uint16_t)NN_MAX_SAMPLES_PER_FRAME);  put_u16(p + 19, 0);
    put_u32(p + 21, bsp_core_clock_hz());
    put_u32(p + 25, nn_ram_bytes());
    put_u32(p + 29, nn_flash_bytes());
    put_u32(p + 33, bench_infer_cycles());
    return 37u;
}

static uint16_t do_samples(uint8_t cmd, uint16_t len)
{
    uint8_t *p = tx_frame + 6;
    uint16_t n, s;
    float x[NN_IN], t[NN_OUT];
    float loss = 0.0f, ae_sum = 0.0f;
    uint32_t t0;

    if (len < 2u) return reply_status(ST_BAD_LEN);
    n = get_u16(rx_payload);
    if (n == 0u || n > NN_MAX_SAMPLES_PER_FRAME || len != 2u + (uint32_t)n * NN_SAMPLE_FLOATS * 4u)
        return reply_status(ST_BAD_LEN);

    t0 = bsp_cycles();
    for (s = 0; s < n; s++) {
        const uint8_t *src = rx_payload + 2u + (uint32_t)s * NN_SAMPLE_FLOATS * 4u;
        memcpy(x, src, NN_IN * 4u);
        memcpy(t, src + NN_IN * 4u, NN_OUT * 4u);
        if (cmd == CMD_TRAIN) {
#if NN_TRAIN_ON_BOARD
            loss += nn_train_sample(x, t);          /* backprop de ESTA muestra; no se guarda */
#else
            return reply_status(ST_NOT_AVAILABLE);
#endif
        } else {
            float ae;
            loss += nn_eval_sample(x, t, &ae);
            ae_sum += ae;
        }
    }
#if NN_TRAIN_ON_BOARD
    if (cmd == CMD_TRAIN) nn_apply_update(n, g_lr);
#endif
    t0 = bsp_cycles() - t0;

    p[0] = ST_OK;
    put_f32(p + 1, loss);
    if (cmd == CMD_TRAIN) { put_u32(p + 5, t0); return 9u; }
    put_f32(p + 5, ae_sum);
    put_u32(p + 9, t0);
    return 13u;
}

static uint16_t do_get_w(uint16_t len)
{
    uint8_t *p = tx_frame + 6;
    uint32_t off;
    uint16_t cnt;
    if (len != 6u) return reply_status(ST_BAD_LEN);
    off = get_u32(rx_payload);
    cnt = get_u16(rx_payload + 4);
    if (cnt == 0u || cnt > GETW_MAX_CHUNK || off + cnt > NN_N_PARAMS) return reply_status(ST_BAD_ARG);
    p[0] = ST_OK;
    put_u32(p + 1, off);
    put_u16(p + 5, cnt);
    memcpy(p + 7, nn_params() + off, (uint32_t)cnt * 4u);
    return (uint16_t)(7u + (uint32_t)cnt * 4u);
}

static uint16_t do_infer(uint16_t len)
{
    uint8_t *p = tx_frame + 6;
    float in[4u * NN_N_JOINTS], u[NN_N_JOINTS];
    uint32_t t0;
    if (len != 16u * NN_N_JOINTS) return reply_status(ST_BAD_LEN);
    memcpy(in, rx_payload, sizeof(in));
    t0 = bsp_cycles();
    nn_infer(in, in + NN_N_JOINTS, in + 2u * NN_N_JOINTS, in + 3u * NN_N_JOINTS, u);
    t0 = bsp_cycles() - t0;
    p[0] = ST_OK;
    memcpy(p + 1, u, sizeof(u));
    put_u32(p + 1 + sizeof(u), t0);
    return (uint16_t)(1u + sizeof(u) + 4u);
}

static uint16_t dispatch(uint8_t cmd, uint16_t len)
{
    switch (cmd) {
    case CMD_PING:
        tx_frame[6] = ST_OK; tx_frame[7] = NN_PROTO_VERSION;
        return 2u;
    case CMD_INFO:
        return do_info();
    case CMD_TRAIN:
    case CMD_EVAL:
        return do_samples(cmd, len);
    case CMD_GET_W:
        return do_get_w(len);
    case CMD_INFER:
        return do_infer(len);
#if NN_TRAIN_ON_BOARD
    case CMD_INIT:
        if (len != 4u) return reply_status(ST_BAD_LEN);
        {
            uint32_t seed = get_u32(rx_payload);
            if (seed == 0u) {
#if NN_WEIGHTS_VALID
                nn_load_from_header();
#else
                return reply_status(ST_BAD_ARG);
#endif
            } else {
                nn_randomize(seed);
            }
        }
        return reply_status(ST_OK);
    case CMD_SET_LR:
        if (len != 4u) return reply_status(ST_BAD_LEN);
        memcpy(&g_lr, rx_payload, 4);
        return reply_status(ST_OK);
    case CMD_SET_W:
        if (len < 6u) return reply_status(ST_BAD_LEN);
        {
            uint32_t off = get_u32(rx_payload);
            uint16_t cnt = get_u16(rx_payload + 4);
            if (len != 6u + (uint32_t)cnt * 4u) return reply_status(ST_BAD_LEN);
            if (nn_set_params(off, rx_payload + 6, cnt) != 0) return reply_status(ST_BAD_ARG);
        }
        return reply_status(ST_OK);
#else
    case CMD_INIT: case CMD_SET_LR: case CMD_SET_W:
        return reply_status(ST_NOT_AVAILABLE);
#endif
    default:
        return reply_status(ST_BAD_CMD);
    }
}

/* ---------------------------------------------------------------- recepcion de una trama */
void nn_proto_poll(void)
{
    uint8_t b, hdr[4], crcb[2];
    uint16_t len, crc;
    uint8_t cmd, seq;

    if (bsp_uart_read(&b, 1, 1000u) == 0u || b != SOF1) return;
    if (bsp_uart_read(&b, 1, 20000u) == 0u || b != SOF2) return;
    if (bsp_uart_read(hdr, 4, 20000u) < 4u) return;
    len = (uint16_t)(hdr[2] | (hdr[3] << 8));
    if (len > NN_PROTO_MAX_PAYLOAD) return;
    if (len != 0u && bsp_uart_read(rx_payload, len, 300000u) < len) return;
    if (bsp_uart_read(crcb, 2, 20000u) < 2u) return;
    crc = crc16(0xFFFFu, hdr, 4u);
    crc = crc16(crc, rx_payload, len);
    if (crc != (uint16_t)(crcb[0] | (crcb[1] << 8))) return;       /* trama corrupta: la PC reintenta */

    cmd = hdr[0];
    seq = hdr[1];
    if (cmd != CMD_PING && last_valid && seq == last_seq) {          /* reintento: reenviar sin re-ejecutar */
        bsp_uart_write(tx_frame, tx_len);
        return;
    }
    send_response(cmd, seq, dispatch(cmd, len));
    last_seq = seq;
    last_valid = (cmd == CMD_PING) ? 0u : 1u;
}
