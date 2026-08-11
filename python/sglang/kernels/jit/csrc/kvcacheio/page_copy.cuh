#pragma once

#include "hicache.cuh"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

namespace {

struct HiCachePageCopyParams {
  void* __restrict__ dst;
  const void* __restrict__ src;
  const int64_t* __restrict__ dst_page_indices;
  const int64_t* __restrict__ src_page_indices;
  uint64_t num_pages;
  uint64_t dst_page_stride_bytes;
  uint64_t src_page_stride_bytes;
  uint64_t copy_offset_bytes;
  uint64_t copy_bytes;
};

__global__ void hicache_page_copy_kernel(const __grid_constant__ HiCachePageCopyParams params) {
  using namespace device;
  using pack_t = uint4;
  constexpr uint64_t kVecBytes = sizeof(pack_t);

  const uint64_t vectors_per_page = params.copy_bytes / kVecBytes;
  const uint64_t total_vectors = params.num_pages * vectors_per_page;
  const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t thread_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

  for (uint64_t linear = thread_id; linear < total_vectors; linear += thread_stride) {
    const uint64_t logical_page = linear / vectors_per_page;
    const uint64_t vector_in_page = linear % vectors_per_page;
    const uint64_t src_page = static_cast<uint64_t>(params.src_page_indices[logical_page]);
    const uint64_t dst_page = static_cast<uint64_t>(params.dst_page_indices[logical_page]);
    const uint64_t inner_offset = params.copy_offset_bytes + vector_in_page * kVecBytes;
    const auto* src = reinterpret_cast<const pack_t*>(
        pointer::offset(params.src, static_cast<int64_t>(src_page * params.src_page_stride_bytes + inner_offset)));
    auto* dst = reinterpret_cast<pack_t*>(
        pointer::offset(params.dst, static_cast<int64_t>(dst_page * params.dst_page_stride_bytes + inner_offset)));
    details::store_nc(dst, details::load_nc(src));
  }
}

}  // namespace

struct HiCachePageCopyKernel {
  static void
  run(const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView dst_page_indices,
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView src_page_indices,
      const int64_t dst_page_stride_bytes,
      const int64_t src_page_stride_bytes,
      const int64_t copy_offset_bytes,
      const int64_t copy_bytes) {
    using namespace host;

    auto B = SymbolicSize{"raw bytes"};
    auto P = SymbolicSize{"num pages"};
    auto device = SymbolicDevice{};

    TensorMatcher({B}).with_strides({1}).with_dtype<uint8_t>().with_device<kDLGPU, kDLGPUHost, kDLCPU>().verify(dst);
    TensorMatcher({-1}).with_strides({1}).with_dtype<uint8_t>().with_device<kDLGPU, kDLGPUHost, kDLCPU>().verify(src);
    TensorMatcher({P})
        .with_strides({1})
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device)
        .verify(dst_page_indices)
        .verify(src_page_indices);

    RuntimeCheck(dst_page_stride_bytes > 0, "HiCache destination page stride must be positive");
    RuntimeCheck(src_page_stride_bytes > 0, "HiCache source page stride must be positive");
    RuntimeCheck(copy_offset_bytes >= 0, "HiCache page copy offset must be non-negative");
    RuntimeCheck(copy_bytes >= 0, "HiCache page copy size must be non-negative");
    RuntimeCheck(
        copy_offset_bytes + copy_bytes <= dst_page_stride_bytes,
        "HiCache page copy range exceeds the destination page envelope");
    RuntimeCheck(
        copy_offset_bytes + copy_bytes <= src_page_stride_bytes,
        "HiCache page copy range exceeds the source page envelope");
    RuntimeCheck(
        dst_page_stride_bytes % 16 == 0 && src_page_stride_bytes % 16 == 0 && copy_offset_bytes % 16 == 0 &&
            copy_bytes % 16 == 0,
        "HiCache raw page copy requires 16-byte aligned sizes");
    if (P.unwrap() == 0 || copy_bytes == 0) {
      return;
    }

    constexpr uint32_t kBlockSize = 256;
    constexpr uint64_t kMaxBlocks = 4096;
    const uint64_t total_vectors = static_cast<uint64_t>(P.unwrap()) * copy_bytes / 16;
    const uint64_t blocks = std::min(div_ceil(total_vectors, static_cast<uint64_t>(kBlockSize)), kMaxBlocks);
    RuntimeCheck(blocks <= std::numeric_limits<uint32_t>::max(), "HiCache page-copy grid exceeds uint32 range");
    const auto params = HiCachePageCopyParams{
        .dst = host::device_accessible_ptr(dst),
        .src = host::device_accessible_ptr(src),
        .dst_page_indices = static_cast<const int64_t*>(dst_page_indices.data_ptr()),
        .src_page_indices = static_cast<const int64_t*>(src_page_indices.data_ptr()),
        .num_pages = static_cast<uint64_t>(P.unwrap()),
        .dst_page_stride_bytes = static_cast<uint64_t>(dst_page_stride_bytes),
        .src_page_stride_bytes = static_cast<uint64_t>(src_page_stride_bytes),
        .copy_offset_bytes = static_cast<uint64_t>(copy_offset_bytes),
        .copy_bytes = static_cast<uint64_t>(copy_bytes),
    };
    LaunchKernel(static_cast<uint32_t>(blocks), kBlockSize, device.unwrap())(hicache_page_copy_kernel, params);
  }
};

}  // namespace sglang
