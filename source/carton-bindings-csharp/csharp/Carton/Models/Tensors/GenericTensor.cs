namespace Carton.Models.Tensors;

using Helpers;
using static Native.CartonBindings;

public class GenericTensor<TType> : ITensor
{
    protected readonly ulong[] Shape;
    protected TType[] Values;

    public GenericTensor(string key, params ulong[] shape)
    {
        Key = key;
        Shape = shape;
        Values = new TType[Elements];
    }

    /// <summary>
    ///     Get the value at specific index
    /// </summary>
    /// <param name="indices"></param>
    /// <returns></returns>
    public TType this[params ulong[] indices]
    {
        get
        {
            var index = TensorHelper.GetIndex(Shape, indices);
            return Values[index];
        }
        set
        {
            var index = TensorHelper.GetIndex(Shape, indices);
            Values[index] = value;
        }
    }

    public TType[] Raw => Values;
    public string Key { get; }

    public ulong Elements => Shape.Aggregate((a, b) => a * b);

    public virtual unsafe CartonTensor* ToCartonTensor()
    {
        var strides = TensorHelper.GetContiguousStride(Shape);

        CartonTensor* tensor;
        fixed (void* dataPtr = Values)
        fixed (ulong* shapePtr = Shape)
        fixed (ulong* stridesPtr = strides)
        {
            carton_tensor_numeric_from_blob(dataPtr, DataTypeHelper.GetDataTypeForType(typeof(TType)), shapePtr, stridesPtr, (ulong)Shape.Length, new FnPtr_VoidPtr_Void(), null, &tensor);
        }

        return tensor;
    }

    /// <summary>
    ///     Set the data in the tensor to the provided values.
    /// </summary>
    /// <param name="data">The provided values.</param>
    public void LoadFromArray(TType[] data)
    {
        Values = data;
    }

    /// <summary>
    ///     Fills the tensor with one value.
    /// </summary>
    /// <param name="value">The value to set in the tensor.</param>
    public void Fill(TType value)
    {
        for (ulong i = 0; i < Elements; i++)
        {
            Values[i] = value;
        }
    }
}