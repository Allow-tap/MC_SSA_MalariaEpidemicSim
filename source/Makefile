CC = mpicc
CFLAGS = -std=c99 -g -Wall -O3
LIBS = -lm

BIN = mal_mpar

all: $(BIN)

mal_mpar: mal_mpar.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
clean:
	$(RM) $(BIN)