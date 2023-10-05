cargo build -p carton-interface-wasm --target wasm32-unknown-unknown
wasm-tools component new ./target/wasm32-unknown-unknown/debug/carton_interface_wasm.wasm \
    -o carton-component.wasm