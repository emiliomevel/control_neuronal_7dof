/* bsp_host.c -- "micro virtual": el mismo firmware compilado en la PC, hablando por stdin/stdout.
 * Sirve para depurar protocolo, entrenamiento y pesos SIN hardware:
 *     make -C firmware/host_sim      y en config.py:  MICRO_URL = "exec:../firmware/host_sim/micro_sim"   */
#include "bsp.h"
#include "nn_user_cfg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

void bsp_init(void)
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

uint32_t bsp_uart_read(uint8_t *dst, uint32_t n, uint32_t timeout_us)
{
    size_t got = fread(dst, 1, n, stdin);   /* bloqueante; el timeout no aplica en el simulador */
    (void)timeout_us;
    if (got == 0 && feof(stdin)) exit(0);   /* la PC cerro el pipe */
    return (uint32_t)got;
}

void bsp_uart_write(const uint8_t *src, uint32_t n)
{
    fwrite(src, 1, n, stdout);
    fflush(stdout);
}

uint32_t bsp_cycles(void)   /* ns de reloj de pared convertidos a "ciclos" del core configurado */
{
    struct timespec ts;
    uint64_t ns;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ns = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    return (uint32_t)((ns * (BSP_CORE_CLOCK_HZ / 1000000ull)) / 1000ull);
}

uint32_t bsp_core_clock_hz(void) { return BSP_CORE_CLOCK_HZ; }
