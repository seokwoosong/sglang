#include <cuda_runtime.h>

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(error) << "\n";
    std::exit(2);
  }
}

} // namespace

int main(int argc, char **argv) {
  bool use_registered_mmap = true;
  bool use_dedicated_stream = true;
  bool use_device_alias = false;
  size_t copy_count = 80;
  size_t copy_bytes = 196608;
  for (int arg = 1; arg < argc; ++arg) {
    const std::string value = argv[arg];
    if (value == "--cuda-host-alloc") {
      use_registered_mmap = false;
    } else if (value == "--default-stream") {
      use_dedicated_stream = false;
    } else if (value == "--device-alias-destination") {
      use_device_alias = true;
    } else if (value == "--copy-count" && arg + 1 < argc) {
      copy_count = std::stoull(argv[++arg]);
    } else if (value == "--copy-bytes" && arg + 1 < argc) {
      copy_bytes = std::stoull(argv[++arg]);
    } else {
      std::cerr << "Unknown argument: " << value << "\n";
      return 2;
    }
  }

  const size_t total_bytes = copy_count * copy_bytes;
  void *host = nullptr;
  if (use_registered_mmap) {
    host = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host == MAP_FAILED) {
      std::cerr << "mmap failed\n";
      return 2;
    }
    check(cudaHostRegister(host, total_bytes, cudaHostRegisterDefault),
          "cudaHostRegister");
  } else {
    check(cudaHostAlloc(&host, total_bytes, cudaHostAllocDefault),
          "cudaHostAlloc");
  }
  std::memset(host, 0, total_bytes);

  void *host_device_alias = nullptr;
  check(cudaHostGetDevicePointer(&host_device_alias, host, 0),
        "cudaHostGetDevicePointer");

  void *device = nullptr;
  check(cudaMalloc(&device, total_bytes), "cudaMalloc");
  check(cudaMemset(device, 0x5a, total_bytes), "cudaMemset");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after initialization");

  cudaStream_t stream = nullptr;
  if (use_dedicated_stream) {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
  }

  std::vector<const void *> sources(copy_count);
  std::vector<const void *> destinations(copy_count);
  std::vector<size_t> sizes(copy_count, copy_bytes);
  auto *destination_base =
      static_cast<char *>(use_device_alias ? host_device_alias : host);
  for (size_t copy = 0; copy < copy_count; ++copy) {
    sources[copy] = static_cast<const char *>(device) + copy * copy_bytes;
    destinations[copy] = destination_base + copy * copy_bytes;
  }

  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attributes.srcLocHint.type = cudaMemLocationTypeDevice;
  attributes.srcLocHint.id = 0;
  attributes.dstLocHint.type = cudaMemLocationTypeHost;
  attributes.dstLocHint.id = 0;
  size_t attributes_index = 0;
#if CUDART_VERSION >= 13000
  const auto enqueue_error = cudaMemcpyBatchAsync(
      const_cast<void **>(destinations.data()), sources.data(), sizes.data(),
      copy_count, &attributes, &attributes_index, 1, stream);
#else
  size_t failed_copy = static_cast<size_t>(-1);
  const auto enqueue_error = cudaMemcpyBatchAsync(
      const_cast<void **>(destinations.data()),
      const_cast<void **>(sources.data()), sizes.data(), copy_count,
      &attributes, &attributes_index, 1, &failed_copy, stream);
#endif
  std::cout << "allocator="
            << (use_registered_mmap ? "cudaHostRegister(mmap)"
                                    : "cudaHostAlloc")
            << " stream=" << (use_dedicated_stream ? "dedicated" : "default")
            << " destination=" << (use_device_alias ? "device_alias" : "host")
            << " alias_differs=" << (host_device_alias != host ? "yes" : "no")
            << " copies=" << copy_count << " copy_bytes=" << copy_bytes
            << " enqueue=" << cudaGetErrorString(enqueue_error) << "\n";
  if (enqueue_error != cudaSuccess) {
    return 2;
  }

  const auto sync_error = cudaStreamSynchronize(stream);
  std::cout << "synchronize=" << cudaGetErrorString(sync_error) << "\n";
  if (sync_error != cudaSuccess) {
    return 2;
  }

  const auto *bytes = static_cast<const unsigned char *>(host);
  for (size_t offset = 0; offset < total_bytes; ++offset) {
    if (bytes[offset] != 0x5a) {
      std::cerr << "verification failed at byte " << offset << "\n";
      return 2;
    }
  }
  std::cout << "verification=passed\n";

  if (use_dedicated_stream) {
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  }
  check(cudaFree(device), "cudaFree");
  if (use_registered_mmap) {
    check(cudaHostUnregister(host), "cudaHostUnregister");
    munmap(host, total_bytes);
  } else {
    check(cudaFreeHost(host), "cudaFreeHost");
  }
  return 0;
}
