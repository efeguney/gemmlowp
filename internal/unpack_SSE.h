// Copyright 2015 Google Inc. All Rights Reserved.
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

// unpack.h: unpacking the result blocks computed by compute.h,
// storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_SSE_H_
#define GEMMLOWP_INTERNAL_UNPACK_SSE_H_

#include <iostream>

namespace gemmlowp {

template <typename BitDepthParams, typename PackedResultType>
struct UnpackResultImpl<BitDepthParams,
                        MatrixMap<std::uint8_t, MapOrder::ColMajor>,
                        PackedResultType> {
  typedef MatrixMap<std::uint8_t, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const PackedResultType& src,
                     int depth, const std::int32_t* lhs_rank_one_update,
                     const std::int32_t* rhs_rank_one_update,
                     std::int32_t lhs_offset, std::int32_t rhs_offset,
                     std::int32_t result_offset, std::int32_t result_mult_int,
                     std::int32_t result_shift) {
    ScopedProfilingLabel label("optimized path (SSE)");
    std::int32_t term_11 = lhs_offset * rhs_offset * depth + result_offset;
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
//     std::cout << "Efe ..." << std::endl;

#if 0
    for (int c = 0; c < dst->cols(); c++) {
      for (int r = 0; r < dst->rows(); r++) {
        // To understand this code, read
        //   doc/low-precision.txt
        //   doc/less-than-8-bit.txt
        // We have 4 terms to sum: xx, x1, 1x, 11.
        // In case of requantization, we first need to scale them back
        // to the original scale, using RoundingMultiplyByConstantFraction.
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_rank_one_update[r];
        std::int32_t raw_1x = rhs_rank_one_update[c];
        std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        // Sum the 4 terms.
        std::int32_t sum = term_xx + term_x1 + term_1x + term_11;
        // Multiply by result_mult_int / (2^result_shift)
        std::int32_t result =
            (sum * result_mult_int + kRoundingTerm) >> result_shift;
        // Clamp to [0..255] and store to destination.
        (*dst)(r, c) = result > 255 ? 255 : result < 0 ? 0 : result;
      }
    }
#endif
  }
};
}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_SSE_H_
