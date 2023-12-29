using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using bottlenoselabs.C2CS.Runtime;
using Carton.Models;
using Carton.Models.Tensors;
using Carton.Native;
using static Carton.Native.CartonBindings;

namespace Carton;

public class CartonFactory : ICartonFactory
{
    private unsafe IList<CartonModel> InnerModels = new List<CartonModel>();

    /// <inheritdoc/>
    public async Task<CartonModel> LoadModelAsync(string urlOrPath)
    {
        var cartonModel = await Task.Run(() =>
        {
            unsafe
            {
                CartonAsyncNotifier* notifier;
                CartonNotifierCallback callback;
                var callbackArg = new CallbackArg
                {
                    Data = (void*)null
                };
                carton_async_notifier_create(&notifier);
                carton_async_notifier_register(notifier, &callback, &callbackArg);

                // Start loading model
                carton_load(CString.FromString(urlOrPath), Unsafe.As<CartonNotifierCallback, CartonLoadCallback>(ref callback), callbackArg);

                CartonStatus status;
                CallbackArg callbackArgOut;
                CartonBindings.Carton* model;
                // Wait for the model to load
                carton_async_notifier_wait(notifier, (void**)&model, &status, &callbackArgOut);
                carton_async_notifier_destroy(notifier);

                return new CartonModel() { InnerCarton = model };
            }
        });

        InnerModels.Add(cartonModel);

        return cartonModel;
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        foreach (var model in InnerModels)
        {
            DestroyCartonModel(model);
        }
    }

    /// <summary>
    /// Destroys the CartronModel and inner pointers.
    /// </summary>
    /// <param name="cartonModel">The CartonModel to destroy.</param>
    private unsafe void DestroyCartonModel(CartonModel cartonModel)
    {
        carton_destroy(cartonModel.InnerCarton);

        foreach (var inferResult in cartonModel.InnerInferResults)
        {
            DestroyInferResult(inferResult);
        }
    }

    /// <summary>
    /// Destroys the InferResult and inner pointers.
    /// </summary>
    /// <param name="inferResult">The InferResult to destroy.</param>
    private unsafe void DestroyInferResult(InferResult inferResult)
    {
        carton_tensormap_destroy(inferResult.InnerTensorMap);

        foreach(var tensor in inferResult.InnerTensors)
        {
            carton_tensor_destroy((CartonTensor*)tensor);
        }
    }
}
