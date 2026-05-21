CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

SRC     = src/tensor.c src/layers.c src/profiler.c src/main.c
OBJ     = $(SRC:.c=.o)
TARGET  = profiler

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) profile_results.csv
