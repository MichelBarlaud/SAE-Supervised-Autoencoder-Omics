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

#ifndef PROJCODE_INCLUDE_L1_BUCKET_FILTER_HPP
#define PROJCODE_INCLUDE_L1_BUCKET_FILTER_HPP

#include <float.h>
#include <math.h>

#include "utils/types.hpp"

namespace proj {
namespace l1 {

void ProjBF(double* y, double* x, const int length, const double a) {
  union DtB* r1 = new DtB[length];
  union DtB* r1_ptr = r1;
  union DtB* r2 = ((y == x) ? (new DtB[length]) : (union DtB*)x);
  union DtB* r2_ptr = r2;
  int plength;
  int illength;
  double p;
  double tau;
  int currentLength;
  int count = 0;
  int t[257] = {0};
  double s[257] = {0.};
  double minS[257];
  double maxS[257];
  union DtB* tmpswap;
  int* tmp = &t[1];
  int bucketSize;
  int start;
  int i;
  int over = 0;
  double* y_ptr = y - 1;
  double* y_ptr_over = y + length;
  illength = length;
  int depth = DATASIZE - 1;

  for (i = 0; i < 257; ++i) {
    minS[i] = DBL_MAX;
    maxS[i] = DBL_MIN;
  }

  start = 0;
  const double i_v = r1[start].val = *(++y_ptr);
  const size_t i_b = r1[start].byte[depth];
  ++tmp[i_b];
  s[i_b] += i_v;
  minS[i_b] = std::min(minS[i_b], i_v);
  maxS[i_b] = std::max(maxS[i_b], i_v);
  p = i_v - a;
  plength = 1;
  i = 1;
  while (++y_ptr != y_ptr_over) {
    if (*y_ptr > p) {
      if ((p += ((r1[i].val = *y_ptr) - p) / (++plength)) <= *y_ptr - a) {
        p = *y_ptr - a;
        plength = 1;
      }
      const double i_v = r1[i].val;
      const size_t i_b = r1[i].byte[depth];
      ++tmp[i_b];
      s[i_b] += i_v;
      minS[i_b] = std::min(minS[i_b], i_v);
      maxS[i_b] = std::max(maxS[i_b], i_v);
      ++i;
    }
  }
  tau = -a;
  illength = i;
  for (depth = DATASIZE - 1; depth >= 0 && over == 0; depth--) {
    for (i = 1; i < 256; ++i) {  // in-place cumulative.
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
      if ((tau + s[i]) / (count + bucketSize) < minS[i]) {
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
    start = illength - 1;
    plength = 1 + count;
    p = (tau + (r1[start]).val) / plength;
    ++tmp[r1[start].byte[depth]];
    s[r1[start].byte[depth]] += r1[start].val;
    minS[r1[start].byte[depth]] = r1[start].val;
    maxS[r1[start].byte[depth]] = r1[start].val;

    for (i = illength - 2; i >= 0; --i) {
      if (r1[i].val > p) {
        const size_t i_b = r1[i].byte[depth];
        const double i_v = r1[i].val;
        p += (r1[i].val - p) / (++plength);
        ++tmp[i_b];
        s[i_b] += i_v;
        minS[i_b] = std::min(minS[i_b], i_v);
        maxS[i_b] = std::max(maxS[i_b], i_v);
      } else {
        r1[i] = r1[--illength];
      }
    }
    depth++;
  }
  tau /= count;
  for (i = 0; i < length; i++) x[i] = (y[i] > tau) ? y[i] - tau : 0.0;
  delete[] r1_ptr;
  if (y == x) delete[] r2_ptr;
}

}  // namespace l1
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1_BUCKET_FILTER_HPP */
