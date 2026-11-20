/* GENERADO por 04_preprocesar.py (2026-11-13 17:31) -- NO editar a mano.
 * Modo: shared   objetivo: directo */
#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#define NN_N_JOINTS   7
#define NN_SHARED     1   /* 1: misma red por articulacion | 0: red completa */
#define NN_IS_RESIDUAL 0   /* 1: la red entrega solo una correccion (u = u_PID + u_NN) */
#define NN_IN         6
#define NN_H1         16
#define NN_H2         16
#define NN_OUT        1
#define NN_X_CLIP     5.0f
#define NN_Y_SCALE    1.5f   /* u [rad/s] = y_norm * NN_Y_SCALE */
#define NN_JOINT_MARGIN 0.03f

#define NN_X_MEAN_INIT   { 0.0582848303f, -0.0024573605f, 0.0572525933f, -0.00381239061f, -0.00103223557f, -0.00135503034f }
#define NN_X_STD_INIT    { 1.21027935f, 0.502153277f, 1.22659278f, 0.385583282f, 0.336102009f, 0.448425263f }
#define NN_U_MAX_INIT    { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f }
#define NN_Q_MIN_INIT    { -3.14159265f, -0.94f, -3.14159265f, -0.68f, -3.14159265f, -3.14159265f, -3.14159265f }
#define NN_Q_MAX_INIT    { 3.14159265f, 1.88f, 3.14159265f, 3.14159265f, 3.14159265f, 0.56f, 3.14159265f }

#endif /* NN_CONFIG_H */
