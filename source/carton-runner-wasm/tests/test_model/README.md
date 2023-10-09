To build the test model, run:
```shell
cargo build --target wasm32-unknown-unknown --release
```
```shell
wasm-tools component new ./target/wasm32-unknown-unknown/release/basic_model.wasm \
  -o model.wasm
```