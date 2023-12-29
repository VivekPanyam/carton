using bottlenoselabs.C2CS.Runtime;
using Carton.Native;
using static Carton.Native.CartonBindings;

namespace Carton;

public class InferResult : IDisposable
{
    public InferResult()
    {

    }

    internal unsafe CartonBindings.CartonTensorMap* InnerTensorMap { get; set; }

    internal unsafe IList<IntPtr> InnerTensors = new List<IntPtr>();

    /// <summary>
    /// Get and remove a string value.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <returns></returns>
    public async Task<string> GetAndRemoveString(string key)
    {
        var resultString = await Task.Run(() =>
        {
            unsafe
            {
                var cartonTensor = GetCartonTensor(key);

                CString valueString;
                ulong valueStringLength;
                carton_tensor_get_string(cartonTensor, 0, &valueString, &valueStringLength);

                return valueString.ToString()[..(int)valueStringLength];
            }
        });

        return resultString;
    }

    /// <summary>
    /// Get and remove a float value.
    /// </summary>
    /// <param name="key">The key of the value.</param>
    /// <returns></returns>
    public async Task<float> GetAndRemoveFloat(string key)
    {
        var resultFloat = await Task.Run(() =>
        {
            unsafe
            {
                var cartonTensor = GetCartonTensor(key);

                float* valueFloat;
                carton_tensor_data(cartonTensor, (void**)&valueFloat);

                return *valueFloat;
            }
        });

        return resultFloat;
    }

    public unsafe void Dispose()
    {
        carton_tensormap_destroy(InnerTensorMap);
        
        foreach(var cartonTensor in InnerTensors) {
            carton_tensor_destroy((CartonTensor*)cartonTensor);
        }
    }

    /// <summary>
    /// Gets the CartonTensor* from the TensorMap.
    /// </summary>
    /// <param name="key">The key for at which the CartonTensor is present.</param>
    /// <returns></returns>
    private unsafe CartonTensor* GetCartonTensor(string key)
    {
        CartonTensor* cartonTensor;
        carton_tensormap_get_and_remove(InnerTensorMap, CString.FromString(key), &cartonTensor);
        InnerTensors.Add((nint)cartonTensor);

        return cartonTensor;
    }

}
