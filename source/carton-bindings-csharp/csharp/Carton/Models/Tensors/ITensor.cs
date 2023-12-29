using Carton.Native;

namespace Carton.Models.Tensors;

public interface ITensor
{
    /// <summary>
    /// Gets the Tensor key.
    /// </summary>
    /// <returns></returns>
    string GetKey();

    /// <summary>
    /// Converts the ITensor to a CartonTensor*.
    /// </summary>
    /// <returns></returns>
    unsafe CartonBindings.CartonTensor* ToCartonTensor();
}
