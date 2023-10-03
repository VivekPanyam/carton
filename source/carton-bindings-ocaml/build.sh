#!/bin/env bash

output=$(cargo build $1 --message-format json | tail -n 2 | head -n 1)
file_names=$(echo "$output" | jq -r ".filenames|.[]")

for name in $file_names; do
    if [[ "$(printf "%s" "$name" | tail -c 2)" == ".a" ]]; then
        cp "$name" "cartonml/lib/libcarton_bindings_ocaml.a"
    else
        cp "$name" "cartonml/lib/dllcarton_bindings_ocaml.so"
    fi
done

cd cartonml
dune build
