using System.Runtime.CompilerServices;
using bottlenoselabs.C2CS.Runtime;
using Carton.Models.Tensors;
using Carton.Native;
using static Carton.Native.CartonBindings;

namespace Carton.Models;

public class CartonModel
{
    public CartonModel()
    {
    }

    internal unsafe CartonBindings.Carton* InnerCarton;

    internal IList<InferResult> InnerInferResults = new List<InferResult>();

    /// <summary>
    /// Infer a model with a list of tensors.
    /// </summary>
    /// <param name="tensors">The tensors to pass to the model.</param>
    /// <returns></returns>
    public async Task<InferResult> Infer(IList<ITensor> tensors)
    {
        var inferResult = await Task.Run(() =>
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

                // Create an input map
                CartonTensorMap* tensorMap;
                carton_tensormap_create(&tensorMap);

                foreach(var tensor in tensors)
                {
                    var cartonTensor = tensor.ToCartonTensor();

                    carton_tensormap_insert(tensorMap, CString.FromString("input"), cartonTensor);
                }

                // Run inference
                callbackArg.Data = null;
                carton_async_notifier_register(notifier, &callback, &callbackArg);
                carton_infer(InnerCarton, tensorMap, Unsafe.As<CartonNotifierCallback, CartonInferCallback>(ref callback), callbackArg);

                // Wait for inference to complete
                CartonTensorMap* outputs;
                CartonStatus status;
                CallbackArg callbackArgOut;
                carton_async_notifier_wait(notifier, (void**)&outputs, &status, &callbackArgOut);
                carton_async_notifier_destroy(notifier);

                var callback_arg_out_infer_result = (nint)callbackArgOut.Data;

                if (status != CartonStatus.CARTON_STATUS_SUCCESS)
                {
                    return new InferResult();
                }

                return new InferResult()
                {
                    InnerTensorMap = outputs
                };
            }
        });

        InnerInferResults.Add(inferResult);

        return inferResult;
    }
}
