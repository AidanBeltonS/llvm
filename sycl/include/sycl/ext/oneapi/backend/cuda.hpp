//==--------- cuda.hpp - SYCL CUDA backend ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend.hpp>
#include <CL/sycl/program.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace cuda {
// Implementation of various "make" functions resides in libsycl.so and thus
// their interface needs to be backend agnostic.
// TODO: remove/merge with similar functions in sycl::detail
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle);

// Construction of SYCL platform.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, platform>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(
    typename sycl::detail::interop<backend::ext_oneapi_cuda, T>::type Interop) {
  return make_platform(reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL platform.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, device>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(
    typename sycl::detail::interop<backend::ext_oneapi_cuda, T>::type Interop) {
  return make_device(reinterpret_cast<pi_native_handle>(Interop));
}


} // namespace cuda
} // namespace oneapi
} // namespace ext

// CUDA platform specialization
template <>
inline backend_return_t<backend::ext_oneapi_cuda, platform>
get_native<backend::ext_oneapi_cuda, platform>(const platform &C) {

  std::vector<device> platform_devices = C.get_devices();
  std::vector<CUdevice> native_devices(platform_devices.size());

  // Get the native CUdevice type for each device in platform
  for (unsigned int i = 0; i < platform_devices.size(); ++i)
    native_devices[i] =
        get_native<backend::ext_oneapi_cuda>(platform_devices[i]);

  return native_devices;
}

// Specialised from include/CL/sycl/backend.hpp
template <>
inline platform make_platform<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, platform> &BackendObject) {
  pi_native_handle NativeHandle =
      detail::pi::cast<pi_native_handle>(&BackendObject);
  return ext::oneapi::cuda::make_platform(NativeHandle);
}


// Specialised from include/CL/sycl/backend.hpp
template <>
inline device make_device<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, device> &BackendObject) {
  pi_native_handle NativeHandle = static_cast<pi_native_handle>(BackendObject);
  return ext::oneapi::cuda::make_device(NativeHandle);
}






} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
