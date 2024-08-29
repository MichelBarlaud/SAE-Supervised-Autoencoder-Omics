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
 * From the code of Laurent Condat: https://lcondat.github.io
 */

#ifndef PROJCODE_INCLUDE_L1INFTY_SORTEDROWS_HPP
#define PROJCODE_INCLUDE_L1INFTY_SORTEDROWS_HPP


#include <numeric>
#include <vector>
#include <algorithm>

#include "utils/Sort.hpp"
#include "utils/print.hpp"

namespace proj {
namespace l1infty {

inline void SortedRows(double* y, double* x, const int nrows, const int ncols,
                       const double C) {
  double *S = new double[nrows * ncols];
  
  std::vector<int> a(nrows, 1);
  std::iota(std::begin(a), std::end(a), 0);
  std::vector<int> k(nrows, 0);

  // sorted permutation.
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      x[i * ncols + j] = y[i * ncols + j];
    }
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
  double theta = (theta_num - C) / theta_den; 
  bool changed = true;

  while (changed) {
    changed = false;
    for (size_t a_i = 0; a_i < a.size(); a_i++) {
      size_t i = a[a_i];
      size_t j = k[i];
      while(j < ncols-1 && (S[i*ncols + j]-theta)/(j+1) < x[i * ncols + j+1]) {
        j++;
      }
      theta_num -= S[i*ncols + k[i]]/(k[i]+1);
      theta_den -= 1./(k[i]+1);
      k[i] = j;
      if (S[i*ncols + ncols-1] < theta) {
        a[a_i] = a.back();
        a.pop_back();
        a_i--;
        continue;
      }
      theta_num += S[i*ncols + k[i]]/(k[i]+1);
      theta_den += 1./(k[i]+1);
    }
    double theta_prime = (theta_num - C) / theta_den; 
    changed |= theta_prime != theta;
    theta = theta_prime;
  }

  double* x_prime = x;
  for (size_t i = 0; i < nrows; i++) {
    if (S[(i+1) * ncols-1] < theta) {
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = 0;
      }
    } else {
      double mu = (S[i*ncols + k[i]] - theta) / (k[i]+1);
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = fmin(mu, y[id]);
      }
    }
  }
  delete[] S;
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_SORTEDROWS_HPP */
