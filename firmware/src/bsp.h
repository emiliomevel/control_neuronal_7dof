/* bsp.h -- capa minima de hardware. Es lo UNICO que cambia entre el micro real y el simulador en PC. */
#ifndef BSP_H
#define BSP_H
#include <stdint.h>

void     bsp_init(void);
/* Lee hasta n bytes esperando como maximo timeout_us microsegundos en total. Devuelve los bytes leidos. */
uint32_t bsp_uart_read(uint8_t *dst, uint32_t n, uint32_t timeout_us);
void     bsp_uart_write(const uint8_t *src, uint32_t n);
uint32_t bsp_cycles(void);            /* contador de ciclos de CPU (32 bits, envuelve) */
uint32_t bsp_core_clock_hz(void);

#endif
