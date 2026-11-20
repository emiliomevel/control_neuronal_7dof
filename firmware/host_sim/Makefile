# Micro virtual para depurar en PC (gcc/MinGW):   make   ->   micro_sim
SRC = ../src/main.c ../src/nn_mlp.c ../src/nn_proto.c bsp_host.c
micro_sim: $(SRC) $(wildcard ../src/*.h)
	$(CC) -O2 -std=gnu99 -Wall -Wextra -DBSP_TARGET_HOST -I../src $(SRC) -lm -o micro_sim
clean:
	rm -f micro_sim micro_sim.exe
