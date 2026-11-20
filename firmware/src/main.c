/* main.c -- lazo principal. En S32 Design Studio ya tienes un main(): copia estas 3 llamadas
 * despues de Clock_Ip_Init / Siul2_Port_Ip_Init / (interrupciones) y deja el for(;;) al final. */
#include "bsp.h"
#include "nn_mlp.h"
#include "nn_proto.h"

int main(void)
{
    bsp_init();          /* reloj de depuracion (DWT), LPUART */
    nn_init();           /* pesos: aleatorios / header / fijos segun nn_user_cfg.h */
    nn_proto_init();
    for (;;) {
        nn_proto_poll(); /* atiende comandos de la PC: entrenar, evaluar, leer/escribir pesos, inferir */
    }
    return 0;
}
