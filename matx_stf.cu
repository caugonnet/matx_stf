////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <matx.h>

template <class I1>
class hello : public matx::BaseOp<hello<I1>> {
private:
  I1 T_;

public:
  hello(I1 T)
      : T_(T) {}

  __device__ inline void operator()(matx::index_t idx)
  {
    T_(idx) = 2*idx + 1;
  }

  __host__ __device__ inline matx::index_t Size(uint32_t i) const  { return T_.Size(i); }
  static inline constexpr __host__ __device__ int32_t Rank() { return I1::Rank(); }
};

int main(int argc, char **argv) {
    auto a = matx::make_tensor<float>({10});
    a.SetVals({1,2,3,4,5,6,7,8,9,10});

    auto b = a;

    printf("You should see the values 1-10 printed\n");
    matx::print(a);

    matx::stfExecutor exec{};

    hello(a).run(exec);
    hello(b).run(exec);

    auto c = matx::make_tensor<float>({10});
    (c = a + b).run(exec);

    exec.sync();

    printf("You should see the 10 odd values printed\n");
    matx::print(a);

    printf("You should see the 10 odd values printed\n");
    matx::print(b);
}
