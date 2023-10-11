// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "carton.hh"

extern "C"
{
#include "carton.h"
}

#include <utility>

namespace carton
{

    Tensor::Tensor(void *handle) : handle_(handle) {}

    Tensor::Tensor(DataType dtype, std::span<uint64_t> shape)
    {
        CartonTensor *out;
        carton_tensor_create(
            // Note: the C++ DataType enum and the C one are in the same order
            static_cast<::DataType>(dtype),
            shape.data(),
            shape.size(),
            &out);

        handle_ = out;
    }

    Tensor::~Tensor()
    {
        if (handle_ != nullptr)
        {
            carton_tensor_destroy(static_cast<CartonTensor *>(handle_));
        }
    }

    Tensor::Tensor(Tensor &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
    Tensor &Tensor::operator=(Tensor &&other)
    {
        std::swap(handle_, other.handle_);
        return *this;
    }

    Result<Tensor> Tensor::from_blob(const void *data,
                                     DataType dtype,
                                     std::span<uint64_t> shape,
                                     std::span<uint64_t> strides,
                                     void (*deleter)(const void *arg),
                                     const void *deleter_arg)
    {
        CartonTensor *out;
        carton_tensor_numeric_from_blob(
            data,
            // Note: the C++ DataType enum and the C one are in the same order
            static_cast<::DataType>(dtype),
            shape.data(),
            strides.data(),
            shape.size(),
            deleter,
            deleter_arg,
            &out);

        return Tensor(static_cast<void *>(out));
    }

    const void *Tensor::data() const
    {
        void *out;
        carton_tensor_data(static_cast<CartonTensor *>(handle_), &out);
        return out;
    }

    DataType Tensor::dtype() const
    {
        ::DataType dtype;
        carton_tensor_dtype(static_cast<CartonTensor *>(handle_), &dtype);

        // Note: the C++ DataType enum and the C one are in the same order
        return static_cast<carton::DataType>(dtype);
    }

    std::span<const uint64_t> Tensor::shape() const
    {
        const uint64_t *ptr;
        uint64_t len;
        carton_tensor_shape(
            static_cast<CartonTensor *>(handle_),
            &ptr,
            &len);

        return {ptr, len};
    }

    std::span<const int64_t> Tensor::strides() const
    {
        const int64_t *ptr;
        uint64_t len;
        carton_tensor_strides(
            static_cast<CartonTensor *>(handle_),
            &ptr,
            &len);

        return {ptr, len};
    }

    void Tensor::set_string(uint64_t index, std::string_view string)
    {
        carton_tensor_set_string_with_strlen(
            static_cast<CartonTensor *>(handle_),
            index,
            string.data(),
            string.size());
    }

    std::string_view Tensor::get_string(uint64_t index) const
    {
        const char *out;
        uint64_t len;
        carton_tensor_get_string(
            static_cast<CartonTensor *>(handle_),
            index,
            &out,
            &len);

        return {out, len};
    }

    TensorMap::TensorMap()
    {
        CartonTensorMap *out;
        carton_tensormap_create(&out);
        handle_ = out;
    }

    TensorMap::TensorMap(void *handle) : handle_(handle) {}

    TensorMap::TensorMap(std::unordered_map<std::string, Tensor> tensors) : TensorMap()
    {
        // Insert all the tensors
        for (auto &[k, v] : tensors)
        {
            insert(k, std::move(v));
        }
    }

    TensorMap::~TensorMap()
    {
        if (handle_ != nullptr)
        {
            carton_tensormap_destroy(static_cast<CartonTensorMap *>(handle_));
        }
    }

    TensorMap::TensorMap(TensorMap &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
    TensorMap &TensorMap::operator=(TensorMap &&other)
    {
        std::swap(handle_, other.handle_);
        return *this;
    }

    void TensorMap::insert(std::string_view key, Tensor value)
    {
        carton_tensormap_insert_with_strlen(
            static_cast<CartonTensorMap *>(handle_),
            key.data(),
            key.size(),
            static_cast<CartonTensor *>(value.handle_));

        // We're taking ownership of the tensor so we need to make sure it doesn't drop
        value.handle_ = nullptr;
    }

    Tensor TensorMap::get_and_remove(std::string_view key)
    {
        CartonTensor *out;
        carton_tensormap_get_and_remove_with_strlen(
            static_cast<CartonTensorMap *>(handle_),
            key.data(),
            key.size(),
            &out);

        return Tensor(out);
    }

    size_t TensorMap::size()
    {
        uint64_t out;
        carton_tensormap_len(static_cast<CartonTensorMap *>(handle_), &out);
        return static_cast<size_t>(out);
    }

    namespace impl
    {
        AsyncNotifierBase::AsyncNotifierBase()
        {
            CartonAsyncNotifier *out;
            carton_async_notifier_create(&out);
            handle_ = out;
        }

        AsyncNotifierBase::~AsyncNotifierBase()
        {
            if (handle_ != nullptr)
            {
                carton_async_notifier_destroy(static_cast<CartonAsyncNotifier *>(handle_));
            }
        }

        AsyncNotifierBase::AsyncNotifierBase(AsyncNotifierBase &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
        AsyncNotifierBase &AsyncNotifierBase::operator=(AsyncNotifierBase &&other)
        {
            std::swap(handle_, other.handle_);
            return *this;
        }

        std::tuple<void *, Status, void *> AsyncNotifierBase::wait()
        {
            void *out;
            CartonStatus status;
            void *user_arg;
            carton_async_notifier_wait(static_cast<CartonAsyncNotifier *>(handle_), &out, &status, &user_arg);

            return {out, static_cast<Status>(status), user_arg};
        }

        std::optional<std::tuple<void *, Status, void *>> AsyncNotifierBase::get()
        {
            void *out;
            CartonStatus status;
            void *user_arg;

            const auto notifier_status = carton_async_notifier_get(static_cast<CartonAsyncNotifier *>(handle_), &out, &status, &user_arg);

            if (notifier_status == CARTON_STATUS_NO_ASYNC_TASKS_READY)
            {
                return {};
            }

            return std::make_optional(std::make_tuple(out, static_cast<Status>(status), user_arg));
        }

        void *AsyncNotifierBase::handle() const
        {
            return handle_;
        }

    } // namespace impl

    Carton::Carton(void *handle) : handle_(handle) {}
    Carton::~Carton()
    {
        if (handle_ != nullptr)
        {
            carton_destroy(static_cast<::Carton *>(handle_));
        }
    }

    Carton::Carton(Carton &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
    Carton &Carton::operator=(Carton &&other)
    {
        std::swap(handle_, other.handle_);
        return *this;
    }

    namespace impl
    {
        struct UserLoadCallbackWrapper
        {
            void (*callback)(Result<Carton>, void *callback_arg);
            void *callback_arg;
        };

        struct UserInferCallbackWrapper
        {
            void (*callback)(Result<TensorMap>, void *callback_arg);
            void *callback_arg;
        };

        // Allows us to access private constructors
        class ImplUtils
        {
        public:
            static void load_future_callback(::Carton *result, CartonStatus status, void *arg)
            {
                const auto promise = static_cast<std::promise<Result<Carton>> *>(arg);
                if (status == CARTON_STATUS_SUCCESS)
                {
                    promise->set_value(Result(Carton(result)));
                }
                else
                {
                    promise->set_value(Result<Carton>(static_cast<Status>(status)));
                }

                delete promise;
            }

            static void infer_future_callback(::CartonTensorMap *result, CartonStatus status, void *arg)
            {
                const auto promise = static_cast<std::promise<Result<TensorMap>> *>(arg);
                if (status == CARTON_STATUS_SUCCESS)
                {
                    promise->set_value(Result(TensorMap(result)));
                }
                else
                {
                    promise->set_value(Result<TensorMap>(static_cast<Status>(status)));
                }

                delete promise;
            }

            static void load_user_callback(::Carton *result, CartonStatus status, void *arg)
            {
                const auto cb = static_cast<UserLoadCallbackWrapper *>(arg);
                if (status == CARTON_STATUS_SUCCESS)
                {
                    cb->callback(Result(Carton(result)), cb->callback_arg);
                }
                else
                {
                    cb->callback(Result<Carton>(static_cast<Status>(status)), cb->callback_arg);
                }

                delete cb;
            }

            static void infer_user_callback(::CartonTensorMap *result, CartonStatus status, void *arg)
            {
                const auto cb = static_cast<UserInferCallbackWrapper *>(arg);
                if (status == CARTON_STATUS_SUCCESS)
                {
                    cb->callback(Result(TensorMap(result)), cb->callback_arg);
                }
                else
                {
                    cb->callback(Result<TensorMap>(static_cast<Status>(status)), cb->callback_arg);
                }

                delete cb;
            }
        };
    } // namespace

    std::future<Result<Carton>> Carton::load(std::string_view url_or_path)
    {
        // SAFETY: delete in the callback above
        const auto promise = new std::promise<Result<Carton>>();
        carton_load_with_strlen(url_or_path.data(), url_or_path.size(), impl::ImplUtils::load_future_callback, promise);

        return promise->get_future();
    }

    std::future<Result<TensorMap>> Carton::infer(TensorMap tensors)
    {
        // SAFETY: delete in the callback above
        const auto promise = new std::promise<Result<TensorMap>>();
        carton_infer(
            static_cast<::Carton *>(handle_),
            static_cast<CartonTensorMap *>(tensors.handle_),
            impl::ImplUtils::infer_future_callback, promise);

        // We're taking ownership of tensors so we need to make sure it doesn't drop
        tensors.handle_ = nullptr;

        return promise->get_future();
    }

    void Carton::load(std::string_view url_or_path,
                      void (*callback)(Result<Carton>, void *callback_arg),
                      void *callback_arg)
    {
        // SAFETY: delete in the callback above
        const auto cb = new impl::UserLoadCallbackWrapper{callback, callback_arg};
        carton_load_with_strlen(url_or_path.data(), url_or_path.size(), impl::ImplUtils::load_user_callback, cb);
    }

    void Carton::infer(TensorMap tensors,
                       void (*callback)(Result<TensorMap>, void *callback_arg),
                       void *callback_arg)
    {
        // SAFETY: delete in the callback above
        const auto cb = new impl::UserInferCallbackWrapper{callback, callback_arg};
        carton_infer(
            static_cast<::Carton *>(handle_),
            static_cast<CartonTensorMap *>(tensors.handle_),
            impl::ImplUtils::infer_user_callback, cb);

        // We're taking ownership of tensors so we need to make sure it doesn't drop
        tensors.handle_ = nullptr;
    }

    void Carton::load(std::string_view url_or_path,
                      AsyncNotifierHandle<Carton> notifier_handle,
                      void *callback_arg)
    {
        CartonNotifierCallback callback;
        carton_async_notifier_register(static_cast<CartonAsyncNotifier *>(notifier_handle.handle_), &callback, &callback_arg);
        carton_load_with_strlen(url_or_path.data(), url_or_path.size(), (CartonLoadCallback)callback, callback_arg);
    }

    void Carton::infer(TensorMap tensors,
                       AsyncNotifierHandle<TensorMap> notifier_handle,
                       void *callback_arg)
    {
        CartonNotifierCallback callback;
        carton_async_notifier_register(static_cast<CartonAsyncNotifier *>(notifier_handle.handle_), &callback, &callback_arg);
        carton_infer(
            static_cast<::Carton *>(handle_),
            static_cast<CartonTensorMap *>(tensors.handle_),
            (CartonInferCallback)callback, callback_arg);

        // We're taking ownership of tensors so we need to make sure it doesn't drop
        tensors.handle_ = nullptr;
    }

    const char *CartonResultException::what() const noexcept
    {
        return "Tried to get a value from an unsuccessful result.";
    }

} // namespace carton