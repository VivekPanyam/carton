use std::any::Any;
use std::collections::HashMap;

use color_eyre::eyre::eyre;
use color_eyre::Result;
use serde::{Deserialize, Serialize};
use wasmtime::{Engine, Instance, Memory, Module, Store};

use carton_runner_interface::types::{Tensor, TensorStorage};

use crate::types::{to_byte_slice};
pub use crate::types::{DType, OutputMetadata};

mod types;

pub struct WASMModelInstance {
    module: Module,
    store: Store<()>,
    instance: Instance,
    out_md: HashMap<String, OutputMetadata>,
}

impl WASMModelInstance {
    pub fn from_bytes(engine: &Engine, bytes: &[u8], out_md: HashMap<String, OutputMetadata>) -> Result<Self> {
        let module = Module::from_binary(engine, bytes)
            .map_err(|e| eyre!(e))?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| eyre!(e))?;;
        Ok(Self {
            module,
            store,
            instance,
            out_md
        })
    }

    pub fn infer(&mut self, tensors: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let infer_fn = self.instance
            .get_func(&mut self.store, "infer")
            .ok_or(eyre!("WASM model missing infer function!"))?;
        tensors.into_iter().try_for_each(
            |(name, tensor)| self.tensor_to_wasm(name.as_str(), tensor)
        )?;
        infer_fn.call(&mut self.store, &[], &mut [])
            .map_err(|e| eyre!(e))?;
        let mut ret = HashMap::new();
        self.out_md.clone().iter().try_for_each(
            |(name, md)| ret.insert(name.clone(), self.wasm_to_tensor(name, md)?)
                .map_or(Ok(()), |_| Err(eyre!("Duplicate output tensor: {}", name)))
        )?;
        Ok(ret)
    }

    fn memory(&mut self) -> Result<Memory> {
        self.instance.get_memory(&mut self.store, "memory")
            .ok_or(eyre!("Module missing memory!"))
    }

    fn var_offset(&mut self, name: &str) -> Result<usize> {
        self.instance.get_global(&mut self.store, name)
            .ok_or(eyre!("Module missing global for parameter {:?}", name))?
            .get(&mut self.store)
            .i32()
            .ok_or(eyre!("Global {:?} is not a pointer!", name))
            .map(|i| i as usize)
    }
    fn tensor_to_wasm(&mut self, name: &str, t: Tensor) -> Result<()> {
        match t {
            Tensor::Float(v) => self.storage_to_wasm(name, v),
            Tensor::Double(v) => self.storage_to_wasm(name, v),
            Tensor::I8(v) => self.storage_to_wasm(name, v),
            Tensor::I16(v) => self.storage_to_wasm(name, v),
            Tensor::I32(v) => self.storage_to_wasm(name, v),
            Tensor::I64(v) => self.storage_to_wasm(name, v),
            Tensor::U8(v) => self.storage_to_wasm(name, v),
            Tensor::U16(v) => self.storage_to_wasm(name, v),
            Tensor::U32(v) => self.storage_to_wasm(name, v),
            Tensor::U64(v) => self.storage_to_wasm(name, v),
            _ => Err(eyre!("Unsupported tensor type"))
        }
    }

    fn storage_to_wasm<T>(&mut self, name: &str, t: TensorStorage<T>) -> Result<()> {
        let slice = to_byte_slice(t.view().to_slice_memory_order().unwrap());
        let offset = self.var_offset(name)?;
        self.memory()?.write(&mut self.store, offset, slice)?;
        Ok(())
    }

    fn wasm_to_tensor(&mut self, name: &str, md: &OutputMetadata) -> Result<Tensor> {
        let (t, buf) = md.make_tensor();
        let offset = self.var_offset(name)?;
        self.memory()?.read(&mut self.store, offset, buf)?;
        Ok(t)
    }
}