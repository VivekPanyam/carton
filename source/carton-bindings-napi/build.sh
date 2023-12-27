#!/bin/bash
set -eux

napi build --platform --js index.cjs --dts tmp.d.ts $@

# Disable typechecking because napi-rs doesn't correctly generate types for
# our variant enum bindings
cat <(echo "// @ts-nocheck") tmp.d.ts > index.d.ts
rm tmp.d.ts