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

#include "carton.hh"

namespace carton
{
    class CartonResultException : public std::exception
    {
    public:
        const char *what() const noexcept override;
    };

    // Impl for Result
    template <typename T>
    Result<T>::Result(T item) : inner_(std::move(item))
    {
    }

    template <typename T>
    Result<T>::Result(Status err) : inner_(std::move(err))
    {
    }

    template <typename T>
    Result<T>::~Result() = default;

    template <typename T>
    Result<T>::Result(Result &&other) : inner_(std::move(other.inner_))
    {
    }

    template <typename T>
    Result<T> &Result<T>::operator=(Result<T> &&other)
    {
        std::swap(inner_, other.inner_);
        return *this;
    }

    template <typename T>
    bool Result<T>::ok() const
    {
        return inner_.index == 0;
    }

    template <typename T>
    T Result<T>::get_or_throw()
    {
        try
        {
            return std::get<T>(std::move(inner_));
        }
        catch (const std::bad_variant_access &ex)
        {
            throw CartonResultException();
        }
    }

    template <typename T>
    Status Result<T>::status()
    {
        if (ok())
        {
            return Status::kSuccess;
        }
        else
        {
            return std::get<1>(inner_);
        }
    }

    // Utility to let us read and write strings more easily
    // This is returned by `TensorAccessor` when indexing string tensors
    template <typename T>
    class TensorStringValue
    {
    private:
        T &tensor_;

        uint64_t index_;

    public:
        TensorStringValue(T &tensor, uint64_t index) : tensor_(tensor), index_(index) {}

        // Assignment of a string type
        void operator=(std::string_view val)
        {
            tensor_.set_string(index_, val);
        }

        // Reading of a string type
        operator std::string_view() const
        {
            return tensor_.get_string(index_);
        }
    };

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const TensorStringValue<T> &v)
    {
        os << std::string_view(v);
        return os;
    }

    // Impl for Tensor
    // Using the accessor methods can be faster when accessing many elements because
    // they avoid making function calls on each element access
    template <typename T, size_t NumDims>
    auto Tensor::accessor()
    {
        // TODO: assert N == ndims
        // TODO: assert data type
        if constexpr (std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>)
        {
            return TensorAccessor<std::string_view, NumDims, Tensor &>(*this, strides());
        }
        else
        {
            return TensorAccessor<T, NumDims, void *>(data(), strides());
        }
    }

    // Using the accessor methods can be faster when accessing many elements because
    // they avoid making function calls on each element access
    template <typename T, size_t NumDims>
    auto Tensor::accessor() const
    {
        // TODO: assert N == ndims
        // TODO: assert data type
        if constexpr (std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>)
        {
            return TensorAccessor<std::string_view, NumDims, const Tensor &>(*this, strides());
        }
        else
        {
            return TensorAccessor<T, NumDims, const void *>(data(), strides());
        }
    }

    template <typename T, typename... Index>
    auto Tensor::at(Index... index) const
    {
        constexpr auto N = sizeof...(Index);
        auto acc = accessor<T, N>();
        return acc.operator[](std::forward<Index>(index)...);
    }

    template <typename T, typename... Index>
    auto Tensor::at(Index... index)
    {
        constexpr auto N = sizeof...(Index);
        auto acc = accessor<T, N>();
        return acc.operator[](std::forward<Index>(index)...);
    }

    // Impl for TensorAccessor
    template <typename T, size_t NumDims, typename DataContainer>
    template <typename... Index>
    auto TensorAccessor<T, NumDims, DataContainer>::operator[](Index... index) const
    {
        constexpr auto num_indices = sizeof...(Index);
        static_assert(NumDims == num_indices, "Incorrect number of indices");

        // Compute the index. This all gets flattened out at compile time
        int i = 0;

        // Basically sets up a dot product of `index` and `strides`
        auto offset = ([&]
                       { return index * strides_[i++]; }() +
                       ...);

        if constexpr (std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>)
        {
            // Handle string tensors separately
            // For convenience, we allow T to be std::string, but we always use `std::string_view`
            // to avoid unnecessary copies.
            return TensorStringValue(data_, offset);
        }
        else
        {
            // Numeric tensors
            static_assert(std::is_arithmetic_v<T>, "accessor() only supports string and numeric tensors");
            return static_cast<const T *>(data_)[offset * sizeof(T)];
        }
    }

    // Impl for AsyncNotifier
    template <typename T>
    AsyncNotifier<T>::AsyncNotifier() : AsyncNotifierBase() {}

    template <typename T>
    AsyncNotifier<T>::~AsyncNotifier() = default;

    template <typename T>
    AsyncNotifier<T>::AsyncNotifier(AsyncNotifier &&other) : AsyncNotifierBase(std::move(other)) {}

    template <typename T>
    AsyncNotifier<T> &AsyncNotifier<T>::operator=(AsyncNotifier &&other)
    {
        AsyncNotifierBase::operator=(std::move(other));
        return *this;
    }

    template <typename T>
    std::pair<Result<T>, void *> AsyncNotifier<T>::wait()
    {
        const auto [result, status, user_arg] = AsyncNotifierBase::wait();
        if (status == Status::kSuccess)
        {
            return std::make_pair(Result(T(result)), user_arg);
        }
        else
        {
            return std::make_pair(Result<T>(status), user_arg);
        }
    }

    template <typename T>
    std::optional<std::pair<Result<T>, void *>> AsyncNotifier<T>::get()
    {
        const auto data = AsyncNotifierBase::get();
        if (data)
        {
            const auto [result, status, user_arg] = data.value();
            if (status == Status::kSuccess)
            {
                return std::make_pair(Result(T(result)), user_arg);
            }
            else
            {
                return std::make_pair(Result<T>(status), user_arg);
            }
        }
        else
        {
            return {};
        }
    }

    template <typename T>
    AsyncNotifierHandle<T> AsyncNotifier<T>::handle() const
    {
        return AsyncNotifierHandle<T>(AsyncNotifierBase::handle());
    };

} // namespace carton