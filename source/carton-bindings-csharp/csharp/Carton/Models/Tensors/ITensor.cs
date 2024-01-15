using Carton.Native;

namespace Carton.Models.Tensors;

public interface ITensor
{
    /// <summary>
    ///     Gets the amount of elements.
    /// </summary>
    ulong Elements { get; }

    /// <summary>
    ///     Gets the Tensor key.
    /// </summary>
    /// <returns></returns>
    string Key { get; }

    /// <summary>
    ///     Converts the ITensor to a CartonTensor*.
    /// </summary>
    /// <returns></returns>
    unsafe CartonBindings.CartonTensor* ToCartonTensor();
}