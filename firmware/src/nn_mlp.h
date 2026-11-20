/* nn_mlp.h -- MLP IN-H1-H2-OUT (ReLU) con inferencia y, opcionalmente, entrenamiento (backprop) en el micro */
#ifndef NN_MLP_H
#define NN_MLP_H

#include <stdint.h>
#include "nn_user_cfg.h"
#include "nn_config.h"

/* Disposicion del vector plano de parametros (identica a nn_numpy.py):
 *   W1[H1][IN] | b1[H1] | W2[H2][H1] | b2[H2] | W3[OUT][H2] | b3[OUT]          */
#define NN_OFF_W1   0u
#define NN_OFF_B1   (NN_OFF_W1 + NN_H1 * NN_IN)
#define NN_OFF_W2   (NN_OFF_B1 + NN_H1)
#define NN_OFF_B2   (NN_OFF_W2 + NN_H2 * NN_H1)
#define NN_OFF_W3   (NN_OFF_B2 + NN_H2)
#define NN_OFF_B3   (NN_OFF_W3 + NN_OUT * NN_H2)
#define NN_N_PARAMS (NN_OFF_B3 + NN_OUT)

#define NN_SAMPLE_FLOATS      (NN_IN + NN_OUT)
#define NN_MAX_SAMPLES_PER_FRAME ((NN_PROTO_MAX_PAYLOAD - 2u) / (NN_SAMPLE_FLOATS * 4u))

/* --- comunes ------------------------------------------------------------- */
void           nn_init(void);                       /* deja los pesos listos segun el modo de compilacion */
int            nn_weights_valid(void);               /* 1 si nn_weights.h trae pesos validos */
const float   *nn_params(void);                     /* puntero al vector de parametros activo */
float          nn_eval_sample(const float *x, const float *t, float *abs_err);   /* devuelve error cuadratico medio */

/* inferencia completa: estado crudo (q, dq, qr, dqr de NN_N_JOINTS articulaciones) -> u [rad/s]
 * incluye preprocesamiento (e, de, normalizacion, saturacion) y capa de seguridad */
void           nn_infer(const float *q, const float *dq, const float *qr, const float *dqr, float *u);

uint32_t       nn_ram_bytes(void);                  /* RAM estatica usada por la red (pesos+grad+optimizador) */
uint32_t       nn_flash_bytes(void);                /* flash usada por constantes de la red (pesos fijos + normalizacion) */

#if NN_TRAIN_ON_BOARD
void           nn_randomize(uint32_t seed);         /* He uniforme con xorshift32 (seed 0 -> semilla por defecto) */
void           nn_load_from_header(void);
float          nn_train_sample(const float *x, const float *t);   /* forward+backward, acumula gradiente; devuelve loss */
void           nn_apply_update(uint32_t n_samples, float lr);     /* un paso del optimizador con el gradiente medio */
void           nn_reset_optimizer(void);
int            nn_set_params(uint32_t offset, const void *src, uint32_t count);   /* src sin requisito de alineacion */
#endif

#endif /* NN_MLP_H */
