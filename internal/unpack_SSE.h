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

  // Non-optimized rounding routine since SSE does not have
  // optimized kernels for less than 8bit input values
template <std::uint32_t numerator, std::uint32_t denominator>
void SSERoundingMultiplyByConstantFraction(__m128i* x) {
  if (numerator == denominator) {
    return; 
  }
  std::int32_t *x_array = reinterpret_cast<int32_t*>(x);
  for (int i=0; i<4; ++i) {
    x_array[i] = RoundingMultiplyByConstantFraction<numerator, denominator>(x_array[i]);
  }
}



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
    std::uint8_t int_array[16];
    // __m128i term_11_xmm = _mm_set1_epi32(term_11);
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));

    __m128i kRoundingTerm_xmm = _mm_set1_epi32(kRoundingTerm);
    // __m128i result_shift_xmm = _mm_set1_epi32(result_shift);
    __m128i result_shift_xmm = _mm_set_epi32(0, 0, 0, result_shift);
    __m128i result_mult_int_xmm = _mm_set1_epi32(result_mult_int);

    for (int c = 0; c < dst->cols(); c++) {
      std::uint8_t* dst_ptr = dst->data(0, c);
      const std::int32_t* src_ptr = src_map.data(0, c);
      int dst_rows_aligned4 = RoundDown<4>(dst->rows());

      // Round raw_1x and find term_1x
      std::int32_t raw_1x = rhs_rank_one_update[c];
      std::int32_t term_1x =
        RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
      // Broadcast term_1x_xmm
      __m128i term_1x_plus_11_xmm = _mm_set1_epi32(term_1x + term_11);

      int r = 0;
      for (; r < dst_rows_aligned4; r+=4, dst_ptr+=4, src_ptr+=4) {
        // Load/round raw_xx_xmm 
        __m128i term_xx_xmm = _mm_loadu_si128((const __m128i*) src_ptr);
        SSERoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax> (&term_xx_xmm);

        // Load/round raw_x1_xmm 
        __m128i term_x1_xmm = _mm_loadu_si128((const __m128i*) &(lhs_rank_one_update[r]));
        SSERoundingMultiplyByConstantFraction<255, kLhsMax> (&term_x1_xmm);

        // Sum 4 terms: term_11 + term_xx + term_x1 + term_1x
        __m128i sum_xmm = _mm_add_epi32(
            _mm_add_epi32(term_1x_plus_11_xmm, term_xx_xmm), 
            term_x1_xmm);

        // Multiply the sum with result_mult_int
        __m128i res1_xmm = _mm_mul_epi32(result_mult_int_xmm, sum_xmm);
        sum_xmm = _mm_shuffle_epi32(sum_xmm, 0x31);
        __m128i res2_xmm = _mm_mul_epi32(result_mult_int_xmm, sum_xmm);
        __m128i res_xmm = _mm_castps_si128(_mm_shuffle_ps(
              _mm_castsi128_ps(res1_xmm), _mm_castsi128_ps(res2_xmm), 0x88));

        // Add kRoundingTerm_xmm and right shift with result_shift_xmm
        // This is equivalent to dividing by 2^result_shift
        res_xmm = _mm_add_epi32(res_xmm, kRoundingTerm_xmm);
        res_xmm = _mm_sra_epi32(res_xmm, result_shift_xmm);
        res_xmm = _mm_shuffle_epi32(res_xmm, 0xd8);

        // _mm_storeu_si128((__m128i*) &int_array[0], res_xmm);

        // Clamp to 0..255 (saturation)
        __m128i temp1_xmm = _mm_packs_epi32(res_xmm, res_xmm);
        res_xmm = _mm_packus_epi16(temp1_xmm, temp1_xmm);

        // only store 32bit 4x8bit values for now
        // _mm_storeu_si128((__m128i*) &int_array[0], res_xmm);
        // dst_ptr[0] = int_array[0];
        // dst_ptr[1] = int_array[1];
        // dst_ptr[2] = int_array[2];
        // dst_ptr[3] = int_array[3];
        _mm_store_ss((float*) &dst_ptr[0], _mm_castsi128_ps(res_xmm));
      }

      for (; r < dst->rows(); r++) {
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
