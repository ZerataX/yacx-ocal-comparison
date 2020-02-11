#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../lib/ocal/ocal_cuda.hpp"
#include "../lib/yacx/include/yacx/main.hpp"

const std::string cudasource = R"(
    extern "C" __global__ void sumArrayOnGPU (int *A , int *B , int *C , int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            C[i] = A[i] + B[i];
        }
    };
)";
const size_t nElements = 16;
const size_t nBytes = nElements * sizeof(int);

struct RandomGenerator {
  int maxValue;
  RandomGenerator(int max) : maxValue(max) {}

  int operator()() { return std::rand() % maxValue; }
};

void yacxSumarray() {
  Device device("Tesla K20c");
  Source source{cudasource};
  Options options{yacx::options::GpuArchitecture(device),
                  yacx::options::FMAD(false)};

  std::vector<int> inA, inB, out;
  inA.resize(nElements);
  inB.resize(nElements);
  out.resize(nElements);

  std::generate(inA.begin(), inA.end(), RandomGenerator(100));
  std::generate(inB.begin(), inB.end(), RandomGenerator(100));

  std::vector<KernelArg> args;
  args.emplace_back(KernelArg{inA.data(), nBytes, false});
  args.emplace_back(KernelArg{inB.data(), nBytes, false});
  args.emplace_back(KernelArg{out.data(), nBytes, true});
  args.emplace_back(KernelArg{nElements});

  dim3 block(device.max_block_dim);
  dim3 grid(1);

  // on device
  source.program("sumArrayOnGPU")
        .compile(options)
        .configure(block, grid)
        .launch(args, device);
  for (int i = 0; i < out.size(); i++) std::cout << out.at(i) << ' ';
  std::cout << std::endl;

  // on host
  std::transform(inA.begin(), inA.end(), inB.begin(),
                 inA.begin(), std::plus<int>());
  for (int i = 0; i < inA.size(); i++) std::cout << inA.at(i) << ' ';
}

void ocalSumarray() {
  ocal::device<CUDA> device(0);

  int minor, major, max_block;
  device.information(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, &max_block);
  device.information(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &major);
  device.information(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &minor);

  ocal::kernel sumarray =
      ocal::kernel(cuda::source(cudasource),
                   std::vector<std::string>{
                       std::string("--gpu-architecture=compute_") +
                           std::to_string(major) + std::to_string(minor),
                       "--fmad=false"});

  std::vector<int> inAdata, inBdata, outdata;
  inAdata.resize(nElements);
  inBdata.resize(nElements);
  outdata.resize(nElements);

  std::generate(inAdata.begin(), inAdata.end(), RandomGenerator(100));
  std::generate(inBdata.begin(), inBdata.end(), RandomGenerator(100));

  ocal::buffer<int> inA(nElements);
  ocal::buffer<int> inB(nElements);
  ocal::buffer<int> out(nElements);
  std::generate(inB.begin(), inB.end(), RandomGenerator(100));
  auto inA_ptr = inA.get_host_memory_ptr();
  auto inB_ptr = inB.get_host_memory_ptr();
  for (int i = 0; i < inA.size(); ++i) {
    inA_ptr[i] = inAdata.at(i);
    inB_ptr[i] = inBdata.at(i);
  }

  // on device
  device(sumarray)(dim3(max_block), dim3(1))(read(inA), read(inB), write(out),
                                             nElements);
  auto out_ptr = out.get_host_memory_ptr();
  for (int i = 0; i < out.size(); i++) std::cout << out_ptr[i] << ' ';
  std::cout << std::endl;
 
  // on host
  std::transform(inAdata.begin(), inAdata.end(), inBdata.begin(),
                 inAdata.begin(), std::plus<int>());
  for (int i = 0; i < inA.size(); i++) std::cout << inAdata[i] << ' ';
}

int main() {
  ocalSumarray();
  yacxSumarray();
}
