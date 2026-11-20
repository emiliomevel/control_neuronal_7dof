#ifndef APP_RING_BUFFER_H_
#define APP_RING_BUFFER_H_

/* ============================================================================
 * BUFFER CIRCULAR + LECTOR DE LÍNEAS (C puro, sin hardware: se prueba en PC)
 *
 * Un solo productor y un solo consumidor (la interrupción escribe, el main lee)
 * es seguro sin deshabilitar interrupciones: cada índice lo escribe solo uno.
 * ============================================================================ */

#include <stdint.h>

typedef struct {
    uint8_t          *data;
    uint16_t          size;     /* capacidad física; se aprovechan size-1 bytes */
    volatile uint16_t head;     /* lo avanza el productor */
    volatile uint16_t tail;     /* lo avanza el consumidor */
} RingBuffer;

void     Ring_Init(RingBuffer *r, uint8_t *storage, uint16_t size);
int      Ring_Push(RingBuffer *r, uint8_t b);     /*  0 = ok, -1 = lleno  */
int      Ring_Pop(RingBuffer *r, uint8_t *b);     /*  0 = ok, -1 = vacío  */
uint16_t Ring_Count(const RingBuffer *r);
uint16_t Ring_Free(const RingBuffer *r);

/* Ensambla líneas terminadas en \n o \r a partir de los bytes del anillo. */
typedef struct {
    char      *buf;             /* almacenamiento de la línea en construcción */
    uint16_t   cap;
    uint16_t   len;
    uint8_t    overflow;        /* la línea actual excedió `cap` y se descartará */
    uint32_t   dropped_lines;   /* líneas descartadas por ser demasiado largas */
} LineReader;

void LineReader_Init(LineReader *lr, char *storage, uint16_t cap);

/* Consume bytes del anillo hasta completar UNA línea. Devuelve 1 y la copia en
 * dst (con '\0', máx. `max`) o 0 si todavía no hay una línea completa. */
int  LineReader_Poll(LineReader *lr, RingBuffer *r, char *dst, uint32_t max);

#endif /* APP_RING_BUFFER_H_ */
