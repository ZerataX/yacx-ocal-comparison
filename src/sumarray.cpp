#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numberic>
#include <string>
#include <vector>

#include "../lib/ocal/ocal.hpp"
#include "../lib/yacx/main.hpp"

const std::string cudasource =
    "extern C global void sumArrayOnGPU (\n" +
    "float *A , float *B , float * C , int * size ) {\n" +
    "int i_inBlock = threadIdx . x + threadIdx . y * blockDim . x + threadIdx "
    ". z * blockDim . y * blockDim . x;\n" +
    "int blockID = blockIdx . x;\n" +
    "int i = i_inBlock + blockID *( blockDim . x * blockDim . y * blockDim . z "
    ");\n" +
    "if (i <=* size ) {\n" + "C [ i ]= A [ i ]+ B [ i ];\n" + "}\n" + "}";
const size_t nElement = 1024;
const size_t nBytes = nElements * sizeof(float);

void yacx() {
  Device dev;
  Source source{cudasource};
  std::vector<float> inA, inB, out;
  inA.resize(SIZE);
  inB.resize(SIZE);
  out.resize(out);

  std::generate(inA.begin(), inA.end(), std::rand());
  std::generate(inB.begin(), inA.end(), std::rand());

  std::vector<KernelArg> args;
  args.emplace_back(KernelArg{inA.data(), nBytes, false});
  args.emplace_back(KernelArg{inB.data(), nBytes, false});
  args.emplace_back(KernelArg{out.data(), nBytes, true});
  args.emplace_back(KernelArg{nElements});

  dim3 block(dev.max_block_dim);  // dim3 block(nElement);
  dim3 grid(1);
  source.program("sumArrayOnGPU").compile().configure(block, grid).launch(args);

  for (int i = 0; i < out.size(); i++) std::cout << out.at(i) << ' ';
}

void ocal {
  // auto device = ocal::get_device<CUDA>();
  ocal::device<CUDA> device;

  ocal::kernel sumarray = cuda::source(cudasource);

  ocal::buffer<float> inA(nBytes);
  ocal::buffer<float> inB(nBytes);
  ocal::buffer<float> out(nBytes);

  std::generate(inA.begin(), inA.end(), std::rand());
  std::generate(inB.begin(), inB.end(), std::rand());

  dev(sumarray(dim3(nElement), dim3(1))(
      read(inA.begin(), inB.end()), read(in.begin(), in.end()),
      write(out.begin, out.end()), nElements));
  for (int i = 0; i < out.size(); i++) std::cout << out.at(i) << ' ';
}

int main() {
  ocal();
  yacx();
}

