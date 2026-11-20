/* nn_mlp.c -- MLP con ReLU: inferencia + backprop por muestra + Adam/SGD (solo si NN_TRAIN_ON_BOARD)
 * Todo en float32, sin malloc, sin dataset en memoria. */
#include "nn_mlp.h"
#include "nn_weights.h"
#include <math.h>
#include <string.h>

#if (NN_TRAIN_ON_BOARD == 0) && (NN_WEIGHTS_VALID == 0)
#error "nn_weights.h no tiene pesos validos. Entrena (NN_TRAIN_ON_BOARD=1), exporta con --export y recompila con NN_TRAIN_ON_BOARD=0."
#endif
#if NN_WEIGHTS_VALID
typedef char nn_weights_size_check[(NN_WEIGHTS_COUNT == NN_N_PARAMS) ? 1 : -1];  /* arquitectura distinta a la de nn_config.h */
#endif

/* ------------------------------------------------------------------ constantes de preprocesamiento / seguridad */
static const float k_x_mean[NN_IN]       = NN_X_MEAN_INIT;
static const float k_x_std[NN_IN]        = NN_X_STD_INIT;
static const float k_u_max[NN_N_JOINTS]  = NN_U_MAX_INIT;
static const float k_q_min[NN_N_JOINTS]  = NN_Q_MIN_INIT;
static const float k_q_max[NN_N_JOINTS]  = NN_Q_MAX_INIT;

/* ------------------------------------------------------------------ almacenamiento de parametros */
#if NN_TRAIN_ON_BOARD
static float    g_params[NN_N_PARAMS];
static float    g_grad[NN_N_PARAMS];
static float    g_m[NN_N_PARAMS];
#if NN_USE_ADAM
static float    g_v[NN_N_PARAMS];
static float    g_b1t = 1.0f, g_b2t = 1.0f;
#endif
static uint32_t g_rng = 0x2545F491u;
#define PARAMS g_params
#else
#define PARAMS NN_WEIGHTS
#endif

const float *nn_params(void) { return PARAMS; }
int nn_weights_valid(void) { return NN_WEIGHTS_VALID; }

static float clampf(float v, float lo, float hi) { return (v < lo) ? lo : ((v > hi) ? hi : v); }

/* ------------------------------------------------------------------ forward */
static void forward(const float *p, const float *x, float *a1, float *a2, float *y)
{
    const float *w1 = p + NN_OFF_W1, *b1 = p + NN_OFF_B1;
    const float *w2 = p + NN_OFF_W2, *b2 = p + NN_OFF_B2;
    const float *w3 = p + NN_OFF_W3, *b3 = p + NN_OFF_B3;
    uint32_t i, j, k;

    for (i = 0; i < NN_H1; i++) {
        float s = b1[i];
        const float *w = w1 + i * NN_IN;
        for (j = 0; j < NN_IN; j++) s += w[j] * x[j];
        a1[i] = (s > 0.0f) ? s : 0.0f;
    }
    for (j = 0; j < NN_H2; j++) {
        float s = b2[j];
        const float *w = w2 + j * NN_H1;
        for (i = 0; i < NN_H1; i++) s += w[i] * a1[i];
        a2[j] = (s > 0.0f) ? s : 0.0f;
    }
    for (k = 0; k < NN_OUT; k++) {
        float s = b3[k];
        const float *w = w3 + k * NN_H2;
        for (j = 0; j < NN_H2; j++) s += w[j] * a2[j];
        y[k] = s;
    }
}

float nn_eval_sample(const float *x, const float *t, float *abs_err)
{
    float a1[NN_H1], a2[NN_H2], y[NN_OUT], se = 0.0f, ae = 0.0f;
    uint32_t k;
    forward(PARAMS, x, a1, a2, y);
    for (k = 0; k < NN_OUT; k++) {
        float d = y[k] - t[k];
        se += d * d;
        ae += fabsf(d);
    }
    *abs_err = ae / (float)NN_OUT;
    return se / (float)NN_OUT;
}

/* ------------------------------------------------------------------ inferencia desde estado crudo */
static void normalize_x(float *x)
{
    uint32_t m;
    for (m = 0; m < NN_IN; m++) x[m] = clampf((x[m] - k_x_mean[m]) / k_x_std[m], -NN_X_CLIP, NN_X_CLIP);
}

void nn_infer(const float *q, const float *dq, const float *qr, const float *dqr, float *u)
{
    float a1[NN_H1], a2[NN_H2], y[NN_OUT];
    uint32_t j;

#if NN_SHARED
    for (j = 0; j < NN_N_JOINTS; j++) {
        float x[NN_IN];
        x[0] = q[j];  x[1] = dq[j];  x[2] = qr[j];  x[3] = dqr[j];
        x[4] = qr[j] - q[j];  x[5] = dqr[j] - dq[j];
        normalize_x(x);
        forward(PARAMS, x, a1, a2, y);
        u[j] = clampf(y[0], -1.0f, 1.0f) * NN_Y_SCALE;
    }
#else
    {
        float x[NN_IN];
        for (j = 0; j < NN_N_JOINTS; j++) {
            float *xj = x + 6u * j;
            xj[0] = q[j];  xj[1] = dq[j];  xj[2] = qr[j];  xj[3] = dqr[j];
            xj[4] = qr[j] - q[j];  xj[5] = dqr[j] - dq[j];
        }
        normalize_x(x);
        forward(PARAMS, x, a1, a2, y);
        for (j = 0; j < NN_N_JOINTS; j++) u[j] = clampf(y[j], -1.0f, 1.0f) * NN_Y_SCALE;
    }
#endif

    /* ---- capa de seguridad: la red NO tiene autoridad ilimitada ---- */
    for (j = 0; j < NN_N_JOINTS; j++) {
        if (!(u[j] == u[j])) u[j] = 0.0f;                       /* NaN -> 0 */
#if !NN_IS_RESIDUAL
        u[j] = clampf(u[j], -k_u_max[j], k_u_max[j]);           /* |u| <= u_max */
        if (q[j] <= k_q_min[j] + NN_JOINT_MARGIN && u[j] < 0.0f) u[j] = 0.0f;   /* limites articulares */
        if (q[j] >= k_q_max[j] - NN_JOINT_MARGIN && u[j] > 0.0f) u[j] = 0.0f;
#endif
    }
}

uint32_t nn_flash_bytes(void)
{
    uint32_t b = (uint32_t)(sizeof(k_x_mean) + sizeof(k_x_std) + sizeof(k_u_max) + sizeof(k_q_min) + sizeof(k_q_max));
#if !NN_TRAIN_ON_BOARD
    b += (uint32_t)sizeof(NN_WEIGHTS);
#endif
    return b;
}

#if NN_TRAIN_ON_BOARD
/* ================================================================== ENTRENAMIENTO EN EL MICRO */

uint32_t nn_ram_bytes(void)
{
    uint32_t b = (uint32_t)(sizeof(g_params) + sizeof(g_grad) + sizeof(g_m));
#if NN_USE_ADAM
    b += (uint32_t)sizeof(g_v);
#endif
    return b;
}

void nn_reset_optimizer(void)
{
    memset(g_grad, 0, sizeof(g_grad));
    memset(g_m, 0, sizeof(g_m));
#if NN_USE_ADAM
    memset(g_v, 0, sizeof(g_v));
    g_b1t = 1.0f;
    g_b2t = 1.0f;
#endif
}

static uint32_t rng_next(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static void init_layer(float *w, uint32_t n, uint32_t fan_in)
{
    uint32_t i;
    float lim = sqrtf(6.0f / (float)fan_in);                    /* He uniforme */
    for (i = 0; i < n; i++) {
        float u01 = (float)(rng_next() >> 8) * (1.0f / 16777216.0f);
        w[i] = (2.0f * u01 - 1.0f) * lim;
    }
}

void nn_randomize(uint32_t seed)
{
    g_rng = (seed != 0u) ? seed : 0x2545F491u;
    memset(g_params, 0, sizeof(g_params));                      /* biases = 0 */
    init_layer(g_params + NN_OFF_W1, NN_H1 * NN_IN, NN_IN);
    init_layer(g_params + NN_OFF_W2, NN_H2 * NN_H1, NN_H1);
    init_layer(g_params + NN_OFF_W3, NN_OUT * NN_H2, NN_H2);
    nn_reset_optimizer();
}

void nn_load_from_header(void)
{
#if NN_WEIGHTS_VALID
    memcpy(g_params, NN_WEIGHTS, sizeof(g_params));
#else
    nn_randomize(1u);
#endif
    nn_reset_optimizer();
}

void nn_init(void)
{
#if NN_INIT_FROM_HEADER && NN_WEIGHTS_VALID
    nn_load_from_header();
#else
    nn_randomize(1u);
#endif
}

int nn_set_params(uint32_t offset, const void *src, uint32_t count)
{
    if (offset + count > NN_N_PARAMS) return -1;
    memcpy(g_params + offset, src, count * sizeof(float));
    if (offset == 0u) nn_reset_optimizer();
    return 0;
}

/* Una muestra: forward + backward. Acumula dL/dtheta en g_grad (NO actualiza pesos). */
float nn_train_sample(const float *x, const float *t)
{
    float a1[NN_H1], a2[NN_H2], y[NN_OUT];
    float dy[NN_OUT], dz2[NN_H2], dz1[NN_H1];
    float loss = 0.0f;
    const float *w2 = g_params + NN_OFF_W2, *w3 = g_params + NN_OFF_W3;
    float *gw1 = g_grad + NN_OFF_W1, *gb1 = g_grad + NN_OFF_B1;
    float *gw2 = g_grad + NN_OFF_W2, *gb2 = g_grad + NN_OFF_B2;
    float *gw3 = g_grad + NN_OFF_W3, *gb3 = g_grad + NN_OFF_B3;
    uint32_t i, j, k, m;

    forward(g_params, x, a1, a2, y);

    for (k = 0; k < NN_OUT; k++) {                              /* L = mean_k (y-t)^2 */
        float d = y[k] - t[k];
        loss += d * d;
        dy[k] = 2.0f * d / (float)NN_OUT;
    }
    loss /= (float)NN_OUT;

    for (k = 0; k < NN_OUT; k++) {                              /* capa de salida */
        gb3[k] += dy[k];
        for (j = 0; j < NN_H2; j++) gw3[k * NN_H2 + j] += dy[k] * a2[j];
    }
    for (j = 0; j < NN_H2; j++) {                               /* retropropagacion a la capa 2 */
        float s = 0.0f;
        for (k = 0; k < NN_OUT; k++) s += w3[k * NN_H2 + j] * dy[k];
        dz2[j] = (a2[j] > 0.0f) ? s : 0.0f;                     /* derivada de ReLU */
    }
    for (j = 0; j < NN_H2; j++) {
        gb2[j] += dz2[j];
        for (i = 0; i < NN_H1; i++) gw2[j * NN_H1 + i] += dz2[j] * a1[i];
    }
    for (i = 0; i < NN_H1; i++) {                               /* retropropagacion a la capa 1 */
        float s = 0.0f;
        for (j = 0; j < NN_H2; j++) s += w2[j * NN_H1 + i] * dz2[j];
        dz1[i] = (a1[i] > 0.0f) ? s : 0.0f;
    }
    for (i = 0; i < NN_H1; i++) {
        gb1[i] += dz1[i];
        for (m = 0; m < NN_IN; m++) gw1[i * NN_IN + m] += dz1[i] * x[m];
    }
    return loss;
}

/* Un paso del optimizador con el gradiente MEDIO de las n muestras acumuladas. */
void nn_apply_update(uint32_t n_samples, float lr)
{
    uint32_t i;
    float inv = 1.0f / (float)n_samples;
#if NN_USE_ADAM
    float c1, c2;
    g_b1t *= NN_ADAM_B1;
    g_b2t *= NN_ADAM_B2;
    c1 = 1.0f / (1.0f - g_b1t);
    c2 = 1.0f / (1.0f - g_b2t);
    for (i = 0; i < NN_N_PARAMS; i++) {
        float g = g_grad[i] * inv;
        g_m[i] = NN_ADAM_B1 * g_m[i] + (1.0f - NN_ADAM_B1) * g;
        g_v[i] = NN_ADAM_B2 * g_v[i] + (1.0f - NN_ADAM_B2) * g * g;
        g_params[i] -= lr * (g_m[i] * c1) / (sqrtf(g_v[i] * c2) + NN_ADAM_EPS);
        g_grad[i] = 0.0f;
    }
#else
    for (i = 0; i < NN_N_PARAMS; i++) {
        float g = g_grad[i] * inv;
        g_m[i] = NN_SGD_MOMENTUM * g_m[i] + g;
        g_params[i] -= lr * g_m[i];
        g_grad[i] = 0.0f;
    }
#endif
}
#else  /* ---- pesos fijos: sin entrenamiento ---- */
uint32_t nn_ram_bytes(void) { return 0u; }
void nn_init(void) { /* los pesos ya estan en flash */ }
#endif
