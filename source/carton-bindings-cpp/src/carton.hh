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

#pragma once

#include <future>
#include <variant>
#include <optional>
#include <unordered_map>
#include <span>
#include <string>
#include <string_view>

namespace carton
{

    enum Status
    {
        // The operation was completed successfully
        kSuccess = 0,

        // There were no async tasks ready
        kNoAsyncTasksReady,
    };

    // Data types of tensors
    enum DataType
    {
        kFloat,
        kDouble,
        kString,
        kI8,
        kI16,
        kI32,
        kI64,
        kU8,
        kU16,
        kU32,
        kU64,
    };

    // A result type that used as the return value of some functions that can fail
    template <typename T>
    class Result
    {
    private:
        std::variant<T, Status> inner_;

    public:
        Result(T item);
        Result(Status err);
        ~Result();

        // Move and move assign constructor
        Result(Result &&);
        Result<T> &operator=(Result<T> &&other);

        // Check if the result is successful
        bool ok() const;

        // Get the contained value
        // Throws `CartonResultException` if the result wasn't successful
        // It's recommended to use `ok()` or `status()` before calling this method
        T get_or_throw();

        // Returns kSuccess if the result is successful
        // Otherwise returns the contained status
        Status status();
    };

    namespace impl
    {
        class ImplUtils;
    } // namespace impl

    // The Carton tensor type
    class Tensor
    {
    private:
        void *handle_;
        Tensor(void *handle);

        friend class TensorMap;

    public:
        // Create a tensor with dtype `dtype` and shape `shape`
        Tensor(DataType dtype, std::span<uint64_t> shape);
        ~Tensor();

        // Delete copy and assign constructors
        Tensor(const Tensor &) = delete;
        Tensor &operator=(const Tensor &other) = delete;

        // Move and move assign constructor
        Tensor(Tensor &&);
        Tensor &operator=(Tensor &&other);

        // Create a numeric tensor by wrapping user-owned data.
        // `deleter` will be called with `deleter_arg` when Carton no longer has references to `data`
        static Result<Tensor> from_blob(const void *data,
                                        DataType dtype,
                                        std::span<uint64_t> shape,
                                        std::span<uint64_t> strides,
                                        void (*deleter)(const void *arg),
                                        const void *deleter_arg);

        // Get a pointer to the underlying tensor data. This only works for numeric tensors.
        // Returns nullptr if not numeric.
        // Note: the returned data pointer is only valid while this Tensor is in scope.
        const void *data() const;

        // Return the data type of the tensor
        DataType dtype() const;

        // Get the shape of the tensor
        // Note: the returned span is only valid while this Tensor is in scope
        std::span<const uint64_t> shape() const;

        // Get the strides of the tensor
        // Note: the returned span is only valid while this Tensor is in scope
        std::span<const int64_t> strides() const;

        // For a string tensor, set a string at a particular (flattened) index
        // This will copy data from the provided string_view.
        // TODO: do some template magic to make this easy to use
        void set_string(uint64_t index, std::string_view string);

        // For a string tensor, get a string at a particular (flattened) index
        // Note: the returned view is only valid until the tensor is modified.
        // TODO: do some template magic to make this easy to use
        std::string_view get_string(uint64_t index) const;
    };

    template <typename T>
    class AsyncNotifier;

    // A map of strings to `Tensor`s
    class TensorMap
    {
    private:
        void *handle_;
        TensorMap(void *handle);

        friend class Carton;
        friend class impl::ImplUtils;
        friend class AsyncNotifier<TensorMap>;

    public:
        TensorMap();

        // Delete copy and assign constructors
        TensorMap(const TensorMap &) = delete;
        TensorMap &operator=(const TensorMap &other) = delete;

        // Move and move assign constructor
        TensorMap(TensorMap &&);
        TensorMap &operator=(TensorMap &&other);

        // Implicit conversion from an unordered_map
        TensorMap(std::unordered_map<std::string, Tensor>);
        ~TensorMap();

        // TODO: Implicit conversion to an unordered_map
        // operator std::unordered_map<std::string, Tensor>();

        // Insert a Tensor into the map
        void insert(std::string_view key, Tensor value);

        // Remove a tensor from the map and return it
        Tensor get_and_remove(std::string_view key);

        // Returns the number of elements in the map
        size_t size();
    };

    // A handle used to associate an async task with a notifier
    template <typename T>
    class AsyncNotifierHandle
    {
    private:
        void *handle_;
        AsyncNotifierHandle(void *handle) : handle_(handle) {}
        friend class Carton;
        friend class AsyncNotifier<T>;
    };

    namespace impl
    {
        // Don't use this directly. Use `AsyncNotifier` below.
        class AsyncNotifierBase
        {
        private:
            void *handle_;

        protected:
            AsyncNotifierBase();
            virtual ~AsyncNotifierBase();

            // Delete copy and assign constructors
            AsyncNotifierBase(const AsyncNotifierBase &) = delete;
            AsyncNotifierBase &operator=(const AsyncNotifierBase &other) = delete;

            // Move and move assign constructor
            AsyncNotifierBase(AsyncNotifierBase &&);
            AsyncNotifierBase &operator=(AsyncNotifierBase &&other);

            // Returns the result of a call along with the user-specified argument
            // Blocks if no tasks are ready yet
            std::tuple<void *, Status, void *> wait();

            // Returns the result of a call along with the user-specified argument
            // Does not block
            std::optional<std::tuple<void *, Status, void *>> get();

            // Note: the returned handle is only valid while this AsyncNotifier instance exists
            void *handle() const;
        };
    } // namespace impl

    // A way to get results of async functions in a less restricted environment than using
    // a callback directly.
    template <typename T>
    class AsyncNotifier final : private impl::AsyncNotifierBase
    {
    public:
        AsyncNotifier();
        ~AsyncNotifier();

        // Delete copy and assign constructors
        AsyncNotifier(const AsyncNotifier &) = delete;
        AsyncNotifier &operator=(const AsyncNotifier &other) = delete;

        // Move and move assign constructor
        AsyncNotifier(AsyncNotifier &&);
        AsyncNotifier &operator=(AsyncNotifier &&other);

        // Returns the result of a call along with the user-specified argument
        // Blocks if no tasks are ready yet
        std::pair<Result<T>, void *> wait();

        // Returns the result of a call along with the user-specified argument
        // Does not block
        std::optional<std::pair<Result<T>, void *>> get();

        // Note: the returned handle is only valid while this AsyncNotifier instance exists
        AsyncNotifierHandle<T> handle() const;
    };

    class Carton
    {
    private:
        void *handle_;

        // Use `Carton::load`
        Carton(void *handle);

        friend class impl::ImplUtils;
        friend class AsyncNotifier<Carton>;

    public:
        ~Carton();

        // Delete copy and copy assign constructors
        Carton(const Carton &) = delete;
        Carton &operator=(const Carton &other) = delete;

        // Move and move assign constructor
        Carton(Carton &&);
        Carton &operator=(Carton &&other);

        // Load a carton
        static std::future<Result<Carton>> load(std::string_view url_or_path);

        // Run inference
        std::future<Result<TensorMap>> infer(TensorMap tensors);

        // Load with a callback
        //
        // IMPORTANT: these callbacks should not block or do CPU-intensive work.
        // Doing so could block Carton's internal event system.
        // If you would like callbacks on a thread without these restrictions, see `AsyncNotifier`
        static void load(std::string_view url_or_path,
                         void (*callback)(Result<Carton>, void *callback_arg),
                         void *callback_arg);

        // Infer with a callback
        //
        // IMPORTANT: these callbacks should not block or do CPU-intensive work.
        // Doing so could block Carton's internal event system.
        // If you would like callbacks on a thread without these restrictions, see `AsyncNotifier`
        void infer(TensorMap tensors,
                   void (*callback)(Result<TensorMap>, void *callback_arg),
                   void *callback_arg);

        // Load with a notifier
        static void load(std::string_view url_or_path,
                         AsyncNotifierHandle<Carton> notifier_handle,
                         void *callback_arg);

        // Infer with a notifier
        void infer(TensorMap tensors,
                   AsyncNotifierHandle<TensorMap> notifier_handle,
                   void *callback_arg);
    };

} // namespace carton

#include "carton_impl.hh"
