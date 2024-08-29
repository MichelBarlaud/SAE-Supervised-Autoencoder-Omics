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

#ifndef PROJCODE_INCLUDE_L1W_BUCKET_HPP
#define PROJCODE_INCLUDE_L1W_BUCKET_HPP

#include <cfloat>  // for DBL_MAX
#include <cstdio>
#include <iostream>

#include "utils/types.hpp"

namespace proj {
namespace l1w {

void ProjWB(double* y, double* w, double* x, const int length, const double a) {
  union DtB* r1 = new union DtB[length];  // malloc(sizeof(union DtB) * length);
  union DtB* r1_ptr = r1;
  union DtB* r2 = ((y == x) ? (new DtB[length]) : (union DtB*)x);
  union DtB* r2_ptr = r2;
  union DtB* tmpswap;
  double *ptrwr1, *ptrwr2, *ptry1, *ptry2;
  double* wr1 = ptrwr1 = new double[length];
  double* wr2 = ptrwr2 = new double[length];
  double* y1 = ptry1 = new double[length];
  double* y2 = ptry2 = new double[length];
  double* wrswap = wr1;
  int illength;
  double tau;
  int currentLength;
  int t[257];
  double s[257];
  double wbs[257];
  double minS[257];
  double maxS[257];
  int* tmp;
  tmp = &t[0];
  tmp++;
  int bucketSize;
  int start;

  double sumWY = 0;
  double Ws = 0;

  int i;
  int over = 0;

  illength = length;
  int depth = 7;

  for (i = 0; i < 257; ++i) {
    t[i] = 0;
    s[i] = 0.;
    wbs[i] = 0.;
    minS[i] = DBL_MAX;
    maxS[i] = DBL_MIN;
  }

  for (i = 0; i < length; ++i) {
    r1[i].val = y[i] / w[i];
    y1[i] = y[i];
    wr1[i] = w[i];
    ++tmp[r1[i].byte[depth]];
    wbs[r1[i].byte[depth]] += wr1[i] * wr1[i];
    s[r1[i].byte[depth]] += y[i] * wr1[i];
    minS[r1[i].byte[depth]] = (minS[r1[i].byte[depth]] < r1[i].val)
                                  ? minS[r1[i].byte[depth]]
                                  : r1[i].val;
    maxS[r1[i].byte[depth]] = (maxS[r1[i].byte[depth]] > r1[i].val)
                                  ? maxS[r1[i].byte[depth]]
                                  : r1[i].val;
  }

  tau = -a;
  illength = length;
  for (depth = 7; depth >= 0; depth--) {
    for (i = 1; i < 256; ++i) {  // Count sort.
      tmp[i] = tmp[i] + tmp[i - 1];
    }
    for (i = 0; i < illength; ++i) {
      r2[t[r1[i].byte[depth]]] = r1[i];
      y2[t[r1[i].byte[depth]]] = y1[i];
      wr2[t[r1[i].byte[depth]]++] = wr1[i];
    }

    tmpswap = r2;  // Swap temporary y/w vector
    r2 = r1;
    r1 = tmpswap;
    wrswap = wr2;
    wr2 = wr1;  // Swap temporary w vector
    wr1 = wrswap;
    wrswap = y2;
    y2 = y1;  // Swap temporary y vector
    y1 = wrswap;
    currentLength = illength;

    for (i = 255; i >= 0; --i) {  // t[i] is the starting point of the i+1
                                  // values (because of the ++ )
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
      r1 += start;
      r2 += start;
      wr1 += start;
      wr2 += start;
      y1 += start;
      y2 += start;
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
    for (i = 0; i < illength; ++i) {
      tmp[r1[i].byte[depth]]++;
      wbs[r1[i].byte[depth]] += wr1[i] * wr1[i];
      s[r1[i].byte[depth]] += y1[i] * wr1[i];
      minS[r1[i].byte[depth]] = (minS[r1[i].byte[depth]] < r1[i].val)
                                    ? minS[r1[i].byte[depth]]
                                    : r1[i].val;
      maxS[r1[i].byte[depth]] = (maxS[r1[i].byte[depth]] > r1[i].val)
                                    ? maxS[r1[i].byte[depth]]
                                    : r1[i].val;
    }
    depth++;
  }
  // printf("%f\n",tau);

  for (i = 0; i < length; i++)
    x[i] = (y[i] > w[i] * tau) ? y[i] - w[i] * tau : 0.0;
  delete[] r1_ptr;
  if (y == x) delete[] r2_ptr;
  delete[] ptry2;
  delete[] ptry1;
  delete[] ptrwr1;
  delete[] ptrwr2;
}

}  // namespace l1w
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1W_BUCKET_HPP */
