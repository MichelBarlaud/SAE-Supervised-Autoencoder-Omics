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

#ifndef PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLSINV_HPP
#define PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLSINV_HPP


#include <algorithm>
#include <numeric>

#include "utils/Sort.hpp"
#include "utils/print.hpp"

namespace proj {
namespace l1infty {

inline double HeapRowsHeapColsInv(double* y, double* x, const int nrows,
                             const int ncols, const double C) {
  ValueCoord *Res = new ValueCoord[nrows];

  std::vector<double> S(nrows, 0.);
  std::vector<double> SS(nrows, 0.);
  std::vector<int> k(nrows, ncols+1);
  std::vector<double*> ends(nrows);

  std::vector<int> lasts_i(nrows,-1);
  std::vector<int> lasts_j(nrows,-1);
  int last_id = 0;



  // row heaps.
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      int id = i * ncols + j;
      x[id] = y[id];
      S[i] += y[id];
    }
    SS[i] = S[i];
  }

  int id;
  // global heap
  for (int i = 0; i < nrows; i++) {
    Res[i].value = -S[i];
    Res[i].coordinates = i;
  }
  ValueCoord* end = Res + nrows;
  std::make_heap(Res, end, ::proj::operator>);

  double theta_num = 0.;
  double theta_den = 0.;
  double theta = 0.;

  int i;
  while (k[i = (*Res).coordinates] > 1) {
    id = i * ncols;

    // for next loop
    lasts_i[last_id] = i;
    lasts_j[last_id] = k[i];
    last_id = (last_id +1) % nrows;

    if (k[i] == ncols+1) {
      k[i] --;
      theta_num += S[i] / k[i];
      theta_den += 1. / k[i];
      theta = (theta_num - C) / theta_den;

      if (theta > SS[i]) {
        theta_num -= S[i] / k[i];
        theta_den -= 1. / k[i];
        theta = (theta_num - C) / theta_den;
        break;
      }

      std::pop_heap(Res, end, ::proj::operator>);
      
      std::make_heap(x + (i * ncols), x + ((i + 1) * ncols), std::greater<double>());
      ends[i] = x + i * ncols + ncols ;

      end[-1].value = -S[i] + x[id]*(k[i]);

      std::push_heap(Res, end, ::proj::operator>);
      continue;
    }

    theta_num -= S[i] / k[i];
    theta_den -= 1. / k[i];

    k[i]--;
    S[i] -= x[id];

    theta_num += S[i] / k[i];
    theta_den += 1. / k[i];

    theta = (theta_num - C) / theta_den;

    if ((S[i] - theta) / k[i] < x[id]) {
      theta_num -= S[i] / k[i];
      theta_den -= 1. / k[i];

      k[i]++;
      S[i] += x[id];

      theta_num += S[i] / k[i];
      theta_den += 1. / k[i];
      theta = (theta_num - C) / theta_den;
      break;
    }

    std::pop_heap(Res, end, ::proj::operator>);

    std::pop_heap(x + id, ends[i]--, std::greater<double>());
    end[-1].value = -S[i] + x[id] * (k[i]);

    std::push_heap(Res, end, ::proj::operator>);
  }
  
  last_id = (last_id - 2 + nrows) % nrows;

  for (size_t i = 0; i < nrows; i++) {
    if (SS[i] < theta) {
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = 0;
      }
    } else {
      double mu = (S[i] - theta) / (k[i]);
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = fmin(mu, y[id]);
      }
    }
  }
  
  delete[] Res;
  return theta;
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLSINV_HPP */
