/* bsp_s32k3.c -- capa de hardware para S32K312 / S32K344 (RTD 7.x, S32 Design Studio).
 *
 * Reutiliza EXACTAMENTE la configuracion y el patron de tu proyecto de la LSTM (que ya funciona):
 *   - LPUART6 (PTA15 / PTA16), instancia 6, config Lpuart_Uart_Ip_xHwConfigPB_6
 *   - callback del driver llamado  Uart_Callback  (Config Tools -> LPUART6 -> Callback), IRQ LPUART6 habilitada
 *   - RX: la recepcion de 1 byte esta SIEMPRE armada; cada byte entra por interrupcion a un anillo (ring_buffer.c)
 *   - TX: AsyncSend por bloques; el callback END_TRANSFER marca "libre"
 *   - OsIf (OSIF_COUNTER_SYSTEM) para los tiempos limite; DWT->CYCCNT para medir ciclos del core
 * Por eso NO hay que tocar el .mex: duplica ese proyecto y reemplaza los fuentes (ver README).
 */
#ifndef BSP_TARGET_HOST

#include "bsp.h"
#include "nn_user_cfg.h"
#include "ring_buffer.h"

#include "Clock_Ip.h"
#include "IntCtrl_Ip.h"
#include "Siul2_Port_Ip.h"
#include "OsIf.h"
#include "Lpuart_Uart_Ip.h"

#include <string.h>

#define BSP_LPUART_INSTANCE   (6U)
#define SIUL2_CONFIG          g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals
#define SIUL2_PINS            NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals

#define BSP_RX_RING_SIZE      2048U
#define BSP_TX_CHUNK          64U
#define BSP_TX_TIMEOUT_US     200000U

static uint8_t          s_rx_byte;
static uint8_t          s_rx_store[BSP_RX_RING_SIZE];
static RingBuffer       s_rx_ring;
static uint8_t          s_tx_chunk[BSP_TX_CHUNK];
static volatile uint8_t s_tx_busy = 0U;

volatile uint32_t g_uart_rx_overflow = 0U;   /* diagnostico (visible en el depurador) */
volatile uint32_t g_uart_rx_errors   = 0U;
volatile uint32_t g_uart_tx_timeouts = 0U;

/* ---- reloj en us sobre OsIf (igual que app_time.c de la LSTM); llamarlo seguido ---- */
static uint32_t bsp_time_us(void)
{
    static uint32_t s_ref = 0U, s_acc_ticks = 0U, s_us = 0U, s_ticks_per_us = 0U;
    uint32_t add;
    if (s_ticks_per_us == 0U) {
        s_ticks_per_us = OsIf_MicrosToTicks(1000U, OSIF_COUNTER_SYSTEM) / 1000U;
        if (s_ticks_per_us == 0U) { s_ticks_per_us = 1U; }
        s_ref = OsIf_GetCounter(OSIF_COUNTER_SYSTEM);
    }
    s_acc_ticks += OsIf_GetElapsed(&s_ref, OSIF_COUNTER_SYSTEM);
    add = s_acc_ticks / s_ticks_per_us;
    s_acc_ticks -= add * s_ticks_per_us;
    s_us += add;
    return s_us;
}

/* ---- contador de ciclos del core (DWT->CYCCNT, Cortex-M7) ---- */
#define BSP_DEMCR       (*(volatile uint32_t *)0xE000EDFCu)
#define BSP_DWT_CTRL    (*(volatile uint32_t *)0xE0001000u)
#define BSP_DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004u)
#define BSP_DWT_LAR     (*(volatile uint32_t *)0xE0001FB0u)

static void dwt_init(void)
{
    BSP_DEMCR |= (1u << 24);          /* TRCENA */
    BSP_DWT_LAR = 0xC5ACCE55u;        /* desbloqueo (M7) */
    BSP_DWT_CYCCNT = 0u;
    BSP_DWT_CTRL |= 1u;               /* CYCCNTENA */
}

uint32_t bsp_cycles(void) { return BSP_DWT_CYCCNT; }
uint32_t bsp_core_clock_hz(void) { return BSP_CORE_CLOCK_HZ; }

static void rx_arm(void)
{
    Lpuart_Uart_Ip_SetRxBuffer(BSP_LPUART_INSTANCE, &s_rx_byte, 1U);
    Lpuart_Uart_Ip_AsyncReceive(BSP_LPUART_INSTANCE, &s_rx_byte, 1U);
}

/* ---- callback del driver: el nombre debe coincidir con el de Config Tools (LPUART6 -> Callback) ---- */
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData)
{
    (void)HwInstance;
    (void)UserData;
    switch (Event) {
    case LPUART_UART_IP_EVENT_RX_FULL:
        if (Ring_Push(&s_rx_ring, s_rx_byte) != 0) { g_uart_rx_overflow++; }
        rx_arm();
        break;
    case LPUART_UART_IP_EVENT_ERROR:
        g_uart_rx_errors++;
        rx_arm();
        break;
    case LPUART_UART_IP_EVENT_END_TRANSFER:
        s_tx_busy = 0U;
        break;
    default:
        break;
    }
}

void bsp_init(void)
{
    /* misma secuencia que el main de la LSTM */
    Clock_Ip_Init(Clock_Ip_aClockConfig);
    Clock_Ip_InitClock(Clock_Ip_aClockConfig);
    while (CLOCK_IP_PLL_LOCKED != Clock_Ip_GetPllStatus()) { __asm volatile ("nop"); }
    Clock_Ip_DistributePll();
    OsIf_Init(NULL_PTR);

    IntCtrl_Ip_Init(&IntCtrlConfig_0);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);
    Siul2_Port_Ip_Init(SIUL2_PINS, SIUL2_CONFIG);

    Ring_Init(&s_rx_ring, s_rx_store, (uint16_t)sizeof(s_rx_store));
    Lpuart_Uart_Ip_Init(BSP_LPUART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);
    Lpuart_Uart_Ip_AsyncReceive(BSP_LPUART_INSTANCE, &s_rx_byte, 1U);
    dwt_init();
    (void)bsp_time_us();              /* inicializa la referencia del reloj */
}

uint32_t bsp_uart_read(uint8_t *dst, uint32_t n, uint32_t timeout_us)
{
    uint32_t got = 0U, t0 = bsp_time_us();
    uint8_t b;
    while (got < n) {
        if (Ring_Pop(&s_rx_ring, &b) == 0) {
            dst[got++] = b;
        } else if ((uint32_t)(bsp_time_us() - t0) > timeout_us) {
            break;
        }
    }
    return got;
}

void bsp_uart_write(const uint8_t *src, uint32_t n)
{
    while (n > 0U) {
        uint32_t c = (n > BSP_TX_CHUNK) ? BSP_TX_CHUNK : n;
        uint32_t t0;
        memcpy(s_tx_chunk, src, c);
        s_tx_busy = 1U;
        t0 = bsp_time_us();
        (void)Lpuart_Uart_Ip_AsyncSend(BSP_LPUART_INSTANCE, s_tx_chunk, c);
        while (s_tx_busy != 0U) {
            if ((uint32_t)(bsp_time_us() - t0) > BSP_TX_TIMEOUT_US) { g_uart_tx_timeouts++; s_tx_busy = 0U; break; }
        }
        src += c;
        n -= c;
    }
}

#endif /* BSP_TARGET_HOST */
