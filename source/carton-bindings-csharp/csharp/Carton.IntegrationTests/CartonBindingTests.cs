using System;
using System.Runtime.CompilerServices;
using bottlenoselabs.C2CS.Runtime;
using Carton.Native;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Carton.Native.CartonBindings;

namespace Carton.IntegrationTests
{
    [TestClass]
    public class CartonBindingTests
    {
        [TestMethod]
        public unsafe void FullTest()
        {
            // Create callback
            CartonAsyncNotifier* notifier;
            CartonNotifierCallback callback;
            var callback_arg = new CallbackArg
            {
                Data = (void*)885
            };
            carton_async_notifier_create(&notifier);
            carton_async_notifier_register(notifier, &callback, &callback_arg);

            // Start loading model
            carton_load(CString.FromString("https://carton.pub/google-research/bert-base-uncased"), Unsafe.As<CartonNotifierCallback, CartonLoadCallback>(ref callback), callback_arg);

            // Wait for the model to load
            CartonBindings.Carton* model;
            CartonStatus status;
            CallbackArg callback_arg_out;
            carton_async_notifier_wait(notifier, (void**)&model, &status, &callback_arg_out);

            Assert.AreEqual(CartonStatus.CARTON_STATUS_SUCCESS, status);

            var callback_arg_out_result = (int)callback_arg_out.Data;
            Assert.AreEqual(885, callback_arg_out_result);

            // Create a tensor
            CartonTensor* tensor;
            ulong shape = 1;
            ulong dims = 1;
            carton_tensor_create(DataType.DATA_TYPE_STRING, &shape, dims, &tensor);
            carton_tensor_set_string(tensor, 0, CString.FromString("Today is a good [MASK]."));

            // Create an input map
            CartonTensorMap* tensors;
            carton_tensormap_create(&tensors);
            carton_tensormap_insert(tensors, CString.FromString("input"), tensor);

            // Run inference
            callback_arg.Data = null;
            carton_async_notifier_register(notifier, &callback, &callback_arg);
            carton_infer(model, tensors, Unsafe.As<CartonNotifierCallback, CartonInferCallback>(ref callback), callback_arg);

            // Wait for inference to complete
            CartonTensorMap* outputs;
            carton_async_notifier_wait(notifier, (void**)&outputs, &status, &callback_arg_out);

            Assert.AreEqual(CartonStatus.CARTON_STATUS_SUCCESS, status);

            var callback_arg_out_infer_result = (nint)callback_arg_out.Data;
            Assert.AreEqual((nint)null, callback_arg_out_infer_result);

            // Get the outputs
            CartonTensor* tokens_out;
            carton_tensormap_get_and_remove(outputs, CString.FromString("tokens"), &tokens_out);

            CartonTensor* scores_out;
            carton_tensormap_get_and_remove(outputs, CString.FromString("scores"), &scores_out);

            CString tokenString;
            ulong tokenStringLength;
            carton_tensor_get_string(tokens_out, 0, &tokenString, &tokenStringLength);

            Assert.AreEqual((ulong)"day".Length, tokenStringLength);

            var token = tokenString.ToString()[..(int)tokenStringLength];
            Console.WriteLine("Got output token: {0}", token);
            Assert.AreEqual("day", token.ToString());

            float* scores_data;
            carton_tensor_data(scores_out, (void**)&scores_data);
            Console.WriteLine("Got output score: {0}", scores_data[0]);

            // Testing that `carton_async_notifier_get` works properly
            void* unused;
            CartonStatus notifier_status = carton_async_notifier_get(notifier, &unused, &status, &callback_arg_out);
            Assert.AreEqual(CartonStatus.CARTON_STATUS_NO_ASYNC_TASKS_READY, notifier_status);

            carton_async_notifier_destroy(notifier);
            carton_destroy(model);
            carton_tensormap_destroy(outputs);
            carton_tensor_destroy(tokens_out);
            carton_tensor_destroy(scores_out);
        }
    }
}