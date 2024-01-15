namespace Carton.Helpers;

public static class TensorHelper
{
    /// <summary>
    ///     Gets the contiguous stride for a specific shape.
    /// </summary>
    /// <param name="shape">The tensor shape.</param>
    /// <returns></returns>
    public static ulong[] GetContiguousStride(ulong[] shape)
    {
        ulong acc = 1;
        var stride = new ulong[shape.Length];
        for (var i = shape.Length - 1; i >= 0; --i)
        {
            stride[i] = acc;
            acc *= shape[i];
        }

        return stride;
    }

    /// <summary>
    ///     Get the index for a shape and NDArray index.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="indices">The indices.</param>
    /// <returns></returns>
    public static ulong GetIndex(ulong[] shape, params ulong[] indices)
    {
        var strides = GetContiguousStride(shape);
        ulong index = 0;
        for (var i = 0; i < indices.Length; ++i)
        {
            index += indices[i] * strides[i];
        }

        return index;
    }
}