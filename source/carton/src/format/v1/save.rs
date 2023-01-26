use crate::error::Result;
use crate::types::{CartonInfo, TensorStorage};

/// Given a path to a filled `model` dir, this function creates a complete carton by saving all the additonal
/// info. Returns a path to the saved file
pub(crate) async fn save<T>(
    info: CartonInfo<T>,
    model_dir_path: &std::path::Path,
) -> Result<std::path::PathBuf>
where
    T: TensorStorage,
{
    todo!()
}
