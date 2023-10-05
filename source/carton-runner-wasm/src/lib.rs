use std::any::{Any, type_name};
use std::collections::HashMap;

use color_eyre::eyre::{ensure, eyre};
use color_eyre::Result;
use serde::{Deserialize, Serialize};
use wasmtime::{Engine, Instance, Module, Store};

use carton_runner_interface::types::{Tensor, TensorStorage};

use crate::types::{OutputMetadata, to_byte_slice};

mod types;

pub struct WASMModelInstance {
    module: Module,
    store: Store<()>,
    instance: Instance,
    out_md: HashMap<String, OutputMetadata>,
}

impl WASMModelInstance {
    pub async fn from_bytes(engine: &Engine, bytes: &[u8], out_md: HashMap<String, OutputMetadata>) -> Result<Self> {
        let module = Module::from_binary(engine, bytes)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        Ok(Self {
            module,
            store,
            instance,
            out_md
        })
    }

    pub fn infer(&mut self, tensors: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let infer_fn = self.instance
            .get_func(&self.store, "infer")
            .ok_or(eyre!("WASM model missing infer function!"))?;
        tensors.iter().for_each(
            |(name, tensor)| self.storage_to_wasm(name, tensor.into())?
        );
        infer_fn.call(&mut self.store, &[], &mut [])?;
        self.out_md.iter().map(|(name, md)| self.wasm_to_tensor(name, md)).collect()
    }

    fn storage_to_wasm<T>(&mut self, name: &str, t: TensorStorage<T>) -> Result<()> {
        ensure!(type_name::<T>() != "String", "String tensors are not supported!");
        let slice = to_byte_slice(t.view().to_slice_memory_order().unwrap());
        let mem = self.instance.get_memory(&mut self.store, name)
            .ok_or(eyre!("Module missing buffer for parameter {:?}", name))?;
        mem.write(&mut self.store, 0, slice)?;
        Ok(())
    }

    fn wasm_to_tensor(&mut self, name: &str, md: &OutputMetadata) -> Result<Tensor> {
        let mem = self.instance.get_memory(&mut self.store, name)
            .ok_or(eyre!("Module missing buffer for parameter {:?}", name))?;
        let (t, buf) = md.make_tensor();
        mem.read(&mut self.store, 0, buf)?;
        Ok(t)
    }
}