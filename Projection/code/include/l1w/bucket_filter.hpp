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
 */

#ifndef PROJCODE_INCLUDE_L1W_BUCKET_FILTER_HPP
#define PROJCODE_INCLUDE_L1W_BUCKET_FILTER_HPP


#include <cfloat>  // for DBL_MAX
#include <cstdio>
#include <iostream>
#include "utils/types.hpp"


namespace proj {
namespace l1w {

void ProjWBF(double* y, double* w, double* x, const int length,
             const double a) {
  // union DtB r1;
  int* perm = new int[2 * length];
  int* perm2 = perm + length;
  int* perswap;
  int* ptrToFree = perm;
  int illength;
  double tau;
  int currentLength;
  int t[257] = {0};  // counter (raddix)
  double s[257] = {0.};
  double wbs[257] = {0.};
  double minS[257];
  double maxS[257];
  int* tmp = &t[1];  // Shift for cumulative (raddix)
  int bucketSize;
  int start;

  double sumWY = 0;
  double Ws = 0;
  double lsumWY = 0;
  double lWs = 0;

  int i;
  int j;
  int over = 0;

  illength = length;
  int depth = DATASIZE - 1;

  for (i = 0; i < 257; ++i) {
    minS[i] = DBL_MAX;
    maxS[i] = DBL_MIN;
  }

  double p = a;

  for (j = i = 0; j < length; ++j) {
    if (y[j] > w[j] * p) {
      perm[i] = j;
      const double wy = w[j] * y[j];
      const double ww = w[j] * w[j];
      sumWY += wy;
      Ws += ww;
      p = (sumWY - a) / Ws;
      if (p <= ((wy - a) / (ww))) {
        sumWY = wy;
        Ws = ww;
        p = (sumWY - a) / Ws;
      }
      const double i_v = y[j] / w[j];
      const size_t i_b = ((const unsigned char*)&i_v)[depth];
      ++tmp[i_b];
      s[i_b] += wy;
      wbs[i_b] += ww;
      minS[i_b] = std::min(minS[i_b], i_v);
      maxS[i_b] = std::max(maxS[i_b], i_v);
      ++i;
    }
  }

  tau = p;
  illength = i;
  sumWY = 0;
  Ws = 0;
  for (depth = 7; depth >= 0; depth--) {
    for (i = 1; i < 256; ++i) {  // Count sort.
      tmp[i] = tmp[i] + tmp[i - 1];
    }
    for (i = 0; i < illength; ++i) {
      const size_t pi = perm[i];
      const DtB d = {y[pi] / w[pi]};
      perm2[t[d.byte[depth]]++] = pi;
    }

    perswap = perm2;  // Swap temporary y/w vector
    perm2 = perm;
    perm = perswap;
    currentLength = illength;

    for (i = 255; i >= 0; --i) {
      start = (i == 0) ? 0 : t[i - 1];
      bucketSize = currentLength - start;
      currentLength -= bucketSize;
      if (bucketSize == 0) {
        continue;
      }
      if (tau > maxS[i]) {  // Best possible remaining value is dominatied: end
        over = 1;
        break;
      }
      if ((sumWY + s[i] - a) / (Ws + wbs[i]) <
          minS[i]) {  // try keeping the min of b
        sumWY += s[i];
        Ws += wbs[i];
        tau = (sumWY - a) / Ws;
        continue;
      }
      perm += start;
      perm2 += start;
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
      wbs[i] = 0.;
      minS[i] = DBL_MAX;
      maxS[i] = DBL_MIN;
    }
    start = illength - 1;
    const size_t p_s = perm[start];
    const double wy = y[p_s] * w[p_s];
    const double ww = w[p_s] * w[p_s];
    lsumWY = wy;
    lWs = ww;
    tau = (lsumWY + sumWY - a) / (Ws + lWs);
    const double i_v = y[p_s] / w[p_s];
    const size_t i_b = ((const unsigned char*)&i_v)[depth];
    ++tmp[i_b];
    s[i_b] += wy;
    wbs[i_b] += ww;
    minS[i_b] = std::min(minS[i_b], i_v);
    maxS[i_b] = std::max(maxS[i_b], i_v);
    for (i = illength - 2; i >= 0; --i) {
      const size_t pi = perm[i];
      if (y[pi] > w[pi] * tau) {
        const double wy = y[pi] * w[pi];
        const double ww = w[pi] * w[pi];
        lsumWY += wy;
        lWs += ww;
        tau = (lsumWY + sumWY - a) / (Ws + lWs);
        const double i_v = y[pi] / w[pi];
        const size_t i_b = ((const unsigned char*)&i_v)[depth];
        s[i_b] += wy;
        wbs[i_b] += ww;
        ++tmp[i_b];
        minS[i_b] = std::min(minS[i_b], i_v);
        maxS[i_b] = std::max(maxS[i_b], i_v);
      } else {
        perm[i] = perm[--illength];
      }
    }
    depth++;
  }

  for (i = 0; i < length; i++) x[i] = std::max(y[i] - w[i] * tau, 0.0);
  delete[] ptrToFree;
}

}  // namespace l1w
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1W_BUCKET_FILTER_HPP */
