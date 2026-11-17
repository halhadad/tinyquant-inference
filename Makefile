CC = gcc
ARCH ?= -march=native
CFLAGS = -std=gnu11 -O3 $(ARCH) -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
LDLIBS = -lm

SRC = src/matrix.c src/quant.c src/mlp.c src/dataset.c
OBJ = $(SRC:src/%.c=build/%.o)
HEADERS = $(wildcard src/*.h) $(wildcard tests/*.h)
TESTS = build/test_matrix build/test_quant build/test_mlp build/test_io

all: tinyquant

tinyquant: $(OBJ) build/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: src/%.c $(HEADERS) | build
	$(CC) $(CFLAGS) -c -o $@ $<

build/test_%: tests/test_%.c $(OBJ) $(HEADERS) | build
	$(CC) $(CFLAGS) -o $@ $< $(OBJ) $(LDLIBS)

build:
	mkdir -p build

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

# Same tests, built with AddressSanitizer and UBSan.
asan:
	mkdir -p build-asan
	@for t in matrix quant mlp io; do \
		$(CC) -std=gnu11 -O1 -g $(ARCH) -fsanitize=address,undefined \
			-o build-asan/test_$$t tests/test_$$t.c $(SRC) $(LDLIBS) && \
		./build-asan/test_$$t || exit 1; \
	done

clean:
	rm -rf build build-asan tinyquant

.PHONY: all test asan clean
