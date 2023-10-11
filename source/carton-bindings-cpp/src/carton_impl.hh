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