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

#ifndef PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLS_HPP
#define PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLS_HPP


#include <algorithm>
#include <numeric>

#include "utils/Sort.hpp"
#include "utils/print.hpp"

namespace proj {
namespace l1infty {

inline void HeapRowsHeapCols(double* y, double* x, const int nrows,
                             const int ncols, const double C) {
  // std::cout << "hepRHC ";
  // ChronoP TS;
  // TS.Start();
  ValueCoord *Res = new ValueCoord[nrows];

  std::vector<double> S(nrows, 0.);
  std::vector<int> k(nrows, 1);
  std::vector<int> a(nrows, 1);
  std::vector<double*> ends(nrows);

  // TS.Stop();
  // std::cout << "data creation " << TS.ellapsed_m_second() << " ";
  // TS.Start();

  // row heaps.
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      x[i * ncols + j] = y[i * ncols + j];
    }
    std::make_heap(x + (i * ncols), x + ((i + 1) * ncols));
  }

  double theta_num = 0.;
  // cumulative sum
  int id;
  for (int i = 0; i < nrows; i++) {
    id = i * ncols;
    theta_num += x[id];
    S[i] = x[id];
    std::pop_heap(x + (id), x + (id + ncols));
    ends[i] = x + id + ncols - 1;

    // global heap
    Res[i].value = -S[i] + x[id];
    Res[i].coordinates = i;
  }
  ValueCoord* end = Res + nrows;
  std::make_heap(Res, end);

  double theta_den = nrows;
  double theta = (theta_num - C) / theta_den;

  // TS.Stop();
  // std::cout << "Sorting " << TS.ellapsed_m_second() << " ";
  // TS.Start();

  while (end != Res) {
    // std::cout << theta << " ";
    int i = (*Res).coordinates;
    // std::cout << i << "\n ";
    id = i * ncols;

    theta_num -= S[i] / (k[i]);
    theta_den -= 1. / (k[i]);

    if (k[i] == ncols) {
      if (S[i] < theta) {
        theta = (theta_num - C) / theta_den;
      }
      std::pop_heap(Res, end--);
      continue;
    }

    if ((S[i] - theta) / (k[i]) >= x[id]) {
      break;
    }

    k[i]++;
    S[i] += x[id];

    if (k[i] < ncols || S[i] >= theta) {
      theta_num += S[i] / (k[i]);
      theta_den += 1. / (k[i]);
    } else {
      theta = (theta_num - C) / theta_den;
      std::pop_heap(Res, end--);
      continue;
    }
    
    theta = (theta_num - C) / theta_den;

    std::pop_heap(Res, end);
    if (k[i] == ncols) {
      end[-1].value = -S[i];
    } else {
      std::pop_heap(x + id, ends[i]--);
      end[-1].value = -S[i] + x[id]*k[i];
    }
    std::push_heap(Res, end);
  }

  for (size_t i = 0; i < nrows; i++) {
    if (k[i] == ncols && S[i] < theta) {
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

  // TS.Stop();
  // std::cout << "Algo " << TS.ellapsed_m_second() << "\n";
  delete[] Res;
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_HEAPROWSHEAPCOLS_HPP */
