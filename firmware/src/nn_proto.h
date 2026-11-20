/* nn_proto.h -- protocolo binario PC <-> micro (ver python/micro_link.py) */
#ifndef NN_PROTO_H
#define NN_PROTO_H
#include <stdint.h>

#define NN_PROTO_VERSION 1u

void nn_proto_init(void);
void nn_proto_poll(void);   /* atiende como mucho una trama; llamar en el lazo principal */

#endif
