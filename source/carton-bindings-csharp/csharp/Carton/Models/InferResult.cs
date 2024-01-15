using static Carton.Native.CartonBindings;

namespace Carton;

using System.Runtime.InteropServices;
using bottlenoselabs.C2CS.Runtime;
using Helpers;
using Models.Tensors;

public class InferResult : IDisposable
{
    internal IDictionary<string, IntPtr> InnerTensors = new Dictionary<string, IntPtr>();

    internal unsafe CartonTensorMap* InnerTensorMap { get; set; }

    public unsafe void Dispose()
    {
        carton_tensormap_destroy(InnerTensorMap);

        foreach (var cartonTensor in InnerTensors)
        {
            carton_tensor_destroy((CartonTensor*)cartonTensor.Value);
        }
    }

    /// <summary>
    ///     Get a scalar string value by it's key.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <returns></returns>
    public async Task<string> GetScalarString(string key)
    {
        var resultString = await Task.Run(() =>
        {
            unsafe
            {
                var cartonTensor = GetCartonTensor(key);
                DataType dataTypeOut;
                carton_tensor_dtype(cartonTensor, &dataTypeOut);

                if (dataTypeOut != DataTypeHelper.GetDataTypeForType(typeof(string)))
                {
                    throw new InvalidOperationException("Provided type is not the same as tensor data type.");
                }

                CString valueString;
                ulong valueStringLength;
                carton_tensor_get_string(cartonTensor, 0, &valueString, &valueStringLength);

                return valueString.ToString()[..(int)valueStringLength];
            }
        });

        return resultString;
    }

    /// <summary>
    ///     Get a string tensor by it's key.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <returns></returns>
    public async Task<StringTensor> GetStringTensor(string key, ulong[] shape)
    {
        var resultTensor = await Task.Run(() =>
        {
            unsafe
            {
                var cartonTensor = GetCartonTensor(key);
                DataType dataTypeOut;
                carton_tensor_dtype(cartonTensor, &dataTypeOut);

                if (dataTypeOut != DataTypeHelper.GetDataTypeForType(typeof(string)))
                {
                    throw new InvalidOperationException("Provided type is not the same as tensor data type.");
                }

                var elements = shape.Aggregate((a, b) => a * b);
                var tensorData = new string[elements];
                for (uint i = 0; i < elements; i++)
                {
                    CString valueString;
                    ulong valueStringLength;
                    carton_tensor_get_string(cartonTensor, i, &valueString, &valueStringLength);
                    tensorData[i] = valueString.ToString()[..(int)valueStringLength];
                }

                return tensorData;
            }
        });

        var tensor = new StringTensor(key, shape);
        tensor.LoadFromArray(resultTensor);
        return tensor;
    }

    /// <summary>
    ///     Get a scalar float by it's key.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <returns></returns>
    public async Task<float> GetScalarFloat(string key)
    {
        return (await GetTensor<float>(key, new ulong[] { 1 }))[0];
    }

    /// <summary>
    ///     Get a tensor by it's key.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <param name="shape">THe shape of the tensor.</param>
    /// <returns></returns>
    public async Task<GenericTensor<TType>> GetTensor<TType>(string key, ulong[] shape)
        where TType : unmanaged
    {
        var resultTensor = await Task.Run(() =>
        {
            unsafe
            {
                var cartonTensor = GetCartonTensor(key);
                DataType dataTypeOut;
                carton_tensor_dtype(cartonTensor, &dataTypeOut);

                if (dataTypeOut != DataTypeHelper.GetDataTypeForType(typeof(TType)))
                {
                    throw new InvalidOperationException("Provided type is not the same as tensor data type.");
                }

                var elements = shape.Aggregate((a, b) => a * b);
                IntPtr* pAllocs;
                carton_tensor_data(cartonTensor, (void**)&pAllocs);

                var typeSize = sizeof(TType);
                var tensorData = new TType[elements];
                for (uint i = 0; i < elements; i++)
                {
                    tensorData[i] = Marshal.PtrToStructure<TType>(IntPtr.Add((IntPtr)pAllocs, (int)(typeSize * i)));
                }

                return tensorData;
            }
        });

        var tensor = new GenericTensor<TType>(key, shape);
        tensor.LoadFromArray(resultTensor);
        return tensor;
    }

    /// <summary>
    ///     Gets the CartonTensor* from the TensorMap.
    /// </summary>
    /// <param name="key">The key for at which the CartonTensor is present.</param>
    /// <returns></returns>
    private unsafe CartonTensor* GetCartonTensor(string key)
    {
        if (InnerTensors.TryGetValue(key, out var tensor))
        {
            return (CartonTensor*)tensor;
        }

        CartonTensor* cartonTensor;
        carton_tensormap_get_and_remove(InnerTensorMap, CString.FromString(key), &cartonTensor);
        InnerTensors.Add(key, (IntPtr)cartonTensor);

        return cartonTensor;
    }
}