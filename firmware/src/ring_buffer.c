#include "ring_buffer.h"
#include <string.h>

#define RING_BARRIER()  __asm__ volatile ("" ::: "memory")   /* orden de escritura datos -> índice */

void Ring_Init(RingBuffer *r, uint8_t *storage, uint16_t size)
{
    r->data = storage;
    r->size = size;
    r->head = 0U;
    r->tail = 0U;
}

int Ring_Push(RingBuffer *r, uint8_t b)
{
    uint16_t next = (uint16_t)(r->head + 1U);
    if (next >= r->size) { next = 0U; }
    if (next == r->tail) { return -1; }          /* lleno */
    r->data[r->head] = b;
    RING_BARRIER();
    r->head = next;
    return 0;
}

int Ring_Pop(RingBuffer *r, uint8_t *b)
{
    uint16_t next;
    if (r->tail == r->head) { return -1; }       /* vacío */
    *b = r->data[r->tail];
    RING_BARRIER();
    next = (uint16_t)(r->tail + 1U);
    if (next >= r->size) { next = 0U; }
    r->tail = next;
    return 0;
}

uint16_t Ring_Count(const RingBuffer *r)
{
    uint16_t h = r->head, t = r->tail;
    return (h >= t) ? (uint16_t)(h - t) : (uint16_t)(r->size - t + h);
}

uint16_t Ring_Free(const RingBuffer *r)
{
    return (uint16_t)(r->size - 1U - Ring_Count(r));
}

void LineReader_Init(LineReader *lr, char *storage, uint16_t cap)
{
    lr->buf = storage;
    lr->cap = cap;
    lr->len = 0U;
    lr->overflow = 0U;
    lr->dropped_lines = 0U;
}

int LineReader_Poll(LineReader *lr, RingBuffer *r, char *dst, uint32_t max)
{
    uint8_t b;
    while (Ring_Pop(r, &b) == 0)
    {
        if ((b == (uint8_t)'\n') || (b == (uint8_t)'\r'))
        {
            if (lr->overflow != 0U)
            {
                lr->dropped_lines++;             /* línea demasiado larga: se descarta entera */
            }
            else if (lr->len > 0U)
            {
                uint32_t n = lr->len;
                if (n > (max - 1U)) { n = max - 1U; }
                memcpy(dst, lr->buf, n);
                dst[n] = '\0';
                lr->len = 0U;
                return 1;
            }
            lr->len = 0U;                        /* línea vacía (p. ej. el \n de un \r\n) */
            lr->overflow = 0U;
        }
        else if (lr->len < lr->cap)
        {
            lr->buf[lr->len++] = (char)b;
        }
        else
        {
            lr->overflow = 1U;
        }
    }
    return 0;
}
