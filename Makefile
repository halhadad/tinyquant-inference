CC = gcc
ARCH ?= -march=native
CFLAGS = -std=gnu11 -O3 $(ARCH) -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
LDLIBS = -lm

SRC = src/matrix.c
OBJ = $(SRC:src/%.c=build/%.o)
HEADERS = $(wildcard src/*.h) $(wildcard tests/*.h)
TESTS = build/test_matrix

all: test

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
	@for t in matrix; do \
		$(CC) -std=gnu11 -O1 -g $(ARCH) -fsanitize=address,undefined \
			-o build-asan/test_$$t tests/test_$$t.c $(SRC) $(LDLIBS) && \
		./build-asan/test_$$t || exit 1; \
	done

clean:
	rm -rf build build-asan tinyquant

.PHONY: all test asan clean
