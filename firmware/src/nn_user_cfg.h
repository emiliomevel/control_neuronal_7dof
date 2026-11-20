/* ============================================================================
 *  nn_user_cfg.h  --  MACROS DE USUARIO (esto es lo único que editas a mano)
 * ========================================================================== */
#ifndef NN_USER_CFG_H
#define NN_USER_CFG_H

/* ---------------------------------------------------------------------------
 *  MODO DE OPERACION
 *    NN_TRAIN_ON_BOARD = 1 : la red SE ENTRENA en el micro.
 *         - pesos en RAM, se inicializan con CMD_INIT (aleatorios) o con nn_weights.h
 *         - la PC envia el dataset muestra por muestra (backprop, nada se guarda)
 *         - al terminar, la PC lee los pesos (CMD_GET_W) y genera nn_weights.h
 *    NN_TRAIN_ON_BOARD = 0 : PESOS FIJOS.
 *         - pesos "const" en flash (nn_weights.h); no existe codigo de entrenamiento,
 *           ni gradientes, ni estado del optimizador (menos RAM y Flash)
 *         - el micro solo hace inferencia (CMD_INFER) y no hay que reentrenar al encender
 * ------------------------------------------------------------------------- */
#ifndef NN_TRAIN_ON_BOARD
#define NN_TRAIN_ON_BOARD      0
#endif
/* Solo con NN_TRAIN_ON_BOARD = 1: 1 = arrancar con los pesos de nn_weights.h
 * (reentrenar / ajustar fino); 0 = arrancar con pesos aleatorios (xorshift32). */
#ifndef NN_INIT_FROM_HEADER
#define NN_INIT_FROM_HEADER      0
#endif
/* ---------------------------------------------------------------------------
 *  OPTIMIZADOR (debe coincidir con OPTIMIZER de config.py)
 *    NN_USE_ADAM = 1 : Adam      NN_USE_ADAM = 0 : SGD con momentum
 * ------------------------------------------------------------------------- */
#ifndef NN_USE_ADAM
#define NN_USE_ADAM      1
#endif
#define NN_ADAM_B1             0.9f
#define NN_ADAM_B2             0.999f
#define NN_ADAM_EPS            1e-8f
#define NN_SGD_MOMENTUM        0.9f
#define NN_LR_DEFAULT          0.003f      /* la PC lo cambia por epoca con CMD_SET_LR */

/* ---------------------------------------------------------------------------
 *  HARDWARE / COMUNICACION
 * ------------------------------------------------------------------------- */
#define BSP_CORE_CLOCK_HZ      120000000UL /* S32K312: 120 MHz ; S32K344: 160 MHz  (ajusta a tu Clock config) */
#define NN_UART_BAUD           115200UL    /* informativo: el baud real se fija en Config Tools (LPUART6); debe coincidir con MICRO_URL */
#define NN_PROTO_MAX_PAYLOAD   4096u       /* bytes maximos de payload por trama */

#endif /* NN_USER_CFG_H */
