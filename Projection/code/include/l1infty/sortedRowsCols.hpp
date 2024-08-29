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

#ifndef PROJCODE_INCLUDE_L1INFTY_SORTEDROWSCOLS_HPP
#define PROJCODE_INCLUDE_L1INFTY_SORTEDROWSCOLS_HPP


#include <numeric>

#include "utils/Sort.hpp"
#include "utils/print.hpp"
#include "utils/types.hpp"

namespace proj {
namespace l1infty {

inline void SortedRowsCols(double* y, double* x, const int nrows,
                           const int ncols, const double C) {

  double *S = new double[nrows * ncols];
  size_t ncols_p1 = (ncols + 1);
  ValueCoord* Res = new ValueCoord[nrows * ncols_p1];

  std::vector<int> k(nrows, 0);
  std::vector<int> a(nrows, 1);

  // sorted permutation.
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      x[i * ncols + j] = y[i * ncols + j];
    }
    auto* xi = x + (i * ncols);
    std::sort(x + (i * ncols), x + ((i+1) * ncols), std::greater<double>());
  }

  double theta_num = 0.;
  // cumulative sum
  for (size_t i = 0; i < nrows; i++) {
    S[i * ncols] = x[i * ncols];
    theta_num += x[i * ncols];
    for (size_t j = 1; j < ncols; j++) {
      S[i * ncols + j] = S[i * ncols + j - 1] + x[i * ncols + j];
    }
  }
  double theta_den = nrows;

  // Residual matrix
  int id_p1 = 0;
  int id = 0;
  for (size_t i = 0; i < nrows; i++) {
    Res[id_p1].value = 0;
    Res[id_p1].coordinates = id_p1;
    id_p1++;
    id++;
    for (size_t j = 1; j < ncols; j++) {
      Res[id_p1].value = -S[id - 1] + x[id] * (j);
      Res[id_p1].coordinates = id_p1;
      id_p1++;
      id++;
    }
    Res[id_p1].value = -S[i * ncols + ncols - 1];
    Res[id_p1].coordinates = id_p1;
    id_p1++;
  }
  std::sort(Res, Res + ncols_p1 * nrows, ::proj::operator>);

  double theta = (theta_num - C) / theta_den;
  bool changed = true;
  int t = nrows;

  while (t < nrows * ncols_p1) {
    int i = Res[t].coordinates / ncols_p1;
    int j = Res[t].coordinates % ncols_p1;
    size_t id = i * ncols + j;

    theta_num -= S[i * ncols + k[i]] / (k[i] + 1);
    theta_den -= 1. / (k[i] + 1);

    if (j == ncols) {
      if (S[i * ncols + ncols - 1] < theta) {
        theta = (theta_num - C) / theta_den;
      }
      t++;
      continue;
    }

    if ((S[id - 1] - theta) / (j) >= x[id]) {
      break;
    }

    k[i] = j;

    theta_num += S[i * ncols + k[i]] / (k[i] + 1);
    theta_den += 1. / (k[i] + 1);

    theta = (theta_num - C) / theta_den;
    t++;
  }

  for (size_t i = 0; i < nrows; i++) {
    if (S[(i + 1) * ncols - 1] < theta) {
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = 0;
      }
    } else {
      double mu = (S[i * ncols + k[i]] - theta) / (k[i] + 1);
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = fmin(mu, y[id]);
      }
    }
  }

  delete[] S;
  delete[] Res;
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_SORTEDROWSCOLS_HPP */
