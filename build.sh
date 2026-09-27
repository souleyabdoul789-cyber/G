#!/usr/bin/env bash
# Compile le coeur C de G en librairie partagee (.so).
# Fonctionne tel quel sur Termux (ARM32/ARM64) et sur Linux x86_64 :
# c'est du C99 portable, sans instruction specifique a une architecture.
#
# Prerequis sur Termux : pkg install clang   (ou gcc si disponible)
#
# Usage : ./build.sh

set -e
cd "$(dirname "$0")"

CC=${CC:-gcc}
if ! command -v "$CC" >/dev/null 2>&1; then
    CC=clang
fi

echo "Compilation avec $CC ..."
"$CC" -shared -fPIC -O2 -Wall \
    -o python/G/libg.so \
    c/tensor.c c/nn.c \
    -lm

echo "OK -> python/G/libg.so"
echo "Teste avec : cd python && python3 -c 'from G import Tensor; print(Tensor([[1,2]]))'"
