#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../lib/ocal/ocal_cuda.hpp"
// #include "../lib/yacx/include/yacx/main.hpp"

const std::string cudasource =
    "extern C global void sumArrayOnGPU (float *A , float *B , float * C , int "
    "* size ) {\n"
    "   int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * "
    "blockDim.y * blockDim.x;\n"
    "   int blockID = blockIdx . x;\n"
    "   int i = i_inBlock + blockID *( blockDim.x * blockDim.y * blockDim.z);\n"
    "   if (i <=* size )\n"
    "      C [ i ]= A [ i ]+ B [ i ];\n"
    "}";
const size_t nElements = 1024;
const size_t nBytes = nElements * sizeof(float);

struct RandomGenerator {
  int maxValue;
  RandomGenerator(int max) : maxValue(max) {}

  float operator()() { return static_cast<float>(rand() % maxValue); }
};

// void yacxSumarray() {
//   Device dev;
//   Source source{cudasource};
//   std::vector<float> inA, inB, out;
//   inA.resize(nElements);
//   inB.resize(nElements);
//   out.resize(out);
//
//   std::generate(inA.begin(), inA.end(), RandomGenerator(100));
//   std::generate(inB.begin(), inA.end(), RandomGenerator(100));
//
//   std::vector<KernelArg> args;
//   args.emplace_back(KernelArg{inA.data(), nBytes, false});
//   args.emplace_back(KernelArg{inB.data(), nBytes, false});
//   args.emplace_back(KernelArg{out.data(), nBytes, true});
//   args.emplace_back(KernelArg{nElements});
//
//   Options options{yacx::options::GpuArchitecture(device),
//                   yacx::options::FMAD(false)};
//
//   dim3 block(dev.max_block_dim);  // dim3 block(nElements);
//   dim3 grid(1);
//   source.program("sumArrayOnGPU")
//         .compile(options)
//         .configure(block, grid)
//         .launch(args);
//
//   for (int i = 0; i < out.size(); i++) std::cout << out.at(i) << ' ';
// }

void ocalSumarray() {
  ocal::device<CUDA> device(0);

  int minor, major, max_block;
  device.information(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, &max_block);
  device.information(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &major);
  device.information(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &minor);

  ocal::kernel sumarray =
      ocal::kernel(cuda::source(cudasource),
                   std::vector<std::string>{std::string("--gpu-architecture=") +
                                                std::to_string(major) +
                                                std::to_string(minor),
                                            "--fmad=false"});

  std::vector<float> inAdata, inBdata, outdata;
  inAdata.resize(nElements);
  inBdata.resize(nElements);
  outdata.resize(nElements);

  std::generate(inAdata.begin(), inAdata.end(), RandomGenerator(100));
  std::generate(inBdata.begin(), inAdata.end(), RandomGenerator(100));

  ocal::buffer<float> inA(nBytes);
  ocal::buffer<float> inB(nBytes);
  ocal::buffer<float> out(nBytes);
  auto inA_ptr = inA.get_host_memory_ptr();
  auto inB_ptr = inB.get_host_memory_ptr();
  auto out_ptr = out.get_host_memory_ptr();
  for (int i = 0; i < inAdata.size(); ++i) inA_ptr[i] = inAdata.at(i);
  for (int i = 0; i < inBdata.size(); ++i) inB_ptr[i] = inBdata.at(i);

  device(sumarray)(dim3(max_block), dim3(1))(read(inA), read(inB), write(out),
                                             nElements);
  for (int i = 0; i < out.size(); i++) std::cout << out_ptr[i] << ' ';
}

int main() {
  ocalSumarray();
  // yacxSumarray();
}
