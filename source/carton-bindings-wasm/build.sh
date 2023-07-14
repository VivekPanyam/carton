#!/bin/bash
set -eux
mkdir -p dist
wasm-pack build --target browser --out-dir dist/web --out-name index --release
wasm-pack build --target nodejs --out-dir dist/node --out-name index --release
cp package.json dist