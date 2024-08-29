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

#ifndef PROJCODE_INCLUDE_UTILS_GEN_HPP
#define PROJCODE_INCLUDE_UTILS_GEN_HPP

#include <limits>
#include <random>
#include "utils/types.hpp"

namespace proj {

inline void FillRandMatrix(double* mt, const int nrows, const int ncols) {
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
      std::size_t id = i * ncols + j;
      mt[id] = ((double)rand()) / std::numeric_limits<int>::max();
    }
  }
}

inline void Transpose(double* y, double* y_t, const int nrows,
                      const int ncols) {
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
      std::size_t id = i * ncols + j;
      std::size_t id_t = j * nrows + i;
      y_t[id_t] = y[id];
    }
  }
}

inline double NormL2(double* y, const int nrows, const int ncols) {
  double sum = 0.;
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
      std::size_t id = i * ncols + j;
      sum += y[id] * y[id];
    }
  }
  return sqrt(sum);
}

inline double NormL2Diff(double* y, double* x, const int nrows,
                         const int ncols) {
  double sum = 0.;
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
      std::size_t id = i * ncols + j;
      double v = y[id] - x[id];
      sum += v * v;
    }
  }
  return sqrt(sum);
}

inline double RowSparsity(double* y, const int nrows, const int ncols) {
  int nb_zeroed = 0;
  for (std::size_t i = 0; i < nrows; i++) {
    double sum = 0.;
    for (std::size_t j = 0; j < ncols; j++) {
      std::size_t id = i * ncols + j;
      sum += fabs(y[id]);
    }
    if (sum <= 1e-6) {
      nb_zeroed++;
    }
  }
  return static_cast<double>(nb_zeroed)/nrows;
}

inline void FillRand3DTensor(double* mt, const int d1, const int d2, const int d3) {
  for (std::size_t i = 0; i < d1; i++) {
    for (std::size_t j = 0; j < d2; j++) {
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        mt[id] = ((double)rand()) / std::numeric_limits<int>::max();
      }
    }
  }
}


}  // namespace proj


#endif /* PROJCODE_INCLUDE_UTILS_GEN_HPP */
