/*
 * Copyright (C) 2024 Guillaume Perez
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PROJCODE_INCLUDE_L1_BUCKET_HPP
#define PROJCODE_INCLUDE_L1_BUCKET_HPP

#include <float.h>
#include <math.h>

#include "utils/types.hpp"

namespace proj {
namespace l1 {

void ProjB(const double* y, double* x, const int length, const double a) {
  union DtB* r1 = new DtB[length];
  union DtB* r1_ptr = r1;
  union DtB* r2 = ((y == x) ? (new DtB[length]) : (union DtB*)x);
  union DtB* r2_ptr = r2;
  int illength;
  double tau;
  int currentLength;
  int count = 0;
  int t[257] = {0};
  double s[257] = {0.};
  double minS[257];
  double maxS[257];
  union DtB* tmpswap;
  int* tmp;
  tmp = &t[0];
  tmp++;
  int bucketSize;
  int start;

  int i;
  int over = 0;

  illength = length;
  int depth = DATASIZE - 1;

  for (i = 0; i < 257; ++i) {
    minS[i] = DBL_MAX;
    maxS[i] = DBL_MIN;
  }

  for (i = 0; i < length; i++) {
    const double i_v = r1[i].val = y[i];
    const size_t i_b = r1[i].byte[depth];
    ++tmp[i_b];
    s[i_b] += i_v;
    minS[i_b] = std::min(minS[i_b], i_v);
    maxS[i_b] = std::max(maxS[i_b], i_v);
  }

  tau = -a;
  illength = length;
  for (depth = DATASIZE - 1; depth >= 0; depth--) {
    for (i = 1; i < 256; ++i) {  // Count sort.
      tmp[i] = tmp[i] + tmp[i - 1];
    }
    for (i = 0; i < illength; ++i) {
      r2[t[r1[i].byte[depth]]++] = r1[i];
    }

    tmpswap = r2;
    r2 = r1;
    r1 = tmpswap;
    currentLength = illength;

    for (i = 255; i >= 0; --i) {  // t[i] is the starting point of the i+1
                                  // values (because of the ++ )
      start = (i == 0) ? 0 : t[i - 1];
      bucketSize = currentLength - start;
      currentLength -= bucketSize;
      if (bucketSize == 0) {
        continue;
      }
      if (tau / count >
          maxS[i]) {  // Best possible remaining value is dominatied: end
        over = 1;
        break;
      }
      if ((tau + s[i]) / (count + bucketSize) <
          minS[i]) {  // try keeping the min of b
        tau += s[i];
        count += bucketSize;
        continue;
      }
      r1 += start;
      r2 += start;
      illength = bucketSize;
      break;
    }
    depth--;
    if (depth < 0 || over == 1 || i < 0) {
      break;
    }
    for (i = 0; i < 257; ++i) {
      t[i] = 0;
      s[i] = 0.;
      minS[i] = DBL_MAX;
      maxS[i] = DBL_MIN;
    }
    for (i = 0; i < illength; ++i) {
      const double i_v = r1[i].val;
      const size_t i_b = r1[i].byte[depth];
      ++tmp[i_b];
      s[i_b] += i_v;
      minS[i_b] = std::min(minS[i_b], i_v);
      maxS[i_b] = std::max(maxS[i_b], i_v);
    }
    depth++;
  }
  tau /= count;
  for (i = 0; i < length; i++) x[i] = (y[i] > tau ? y[i] - tau : 0.0);
  delete[] r1_ptr;
  if (y == x) delete[] r2_ptr;
}

}  // namespace l1
}  // namespace proj

#endif /* PROJCODE_INCLUDE_L1_BUCKET_HPP */
