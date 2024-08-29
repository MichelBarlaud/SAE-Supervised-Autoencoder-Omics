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

#ifndef PROJCODE_INCLUDE_L11_BILEVEL11_HPP
#define PROJCODE_INCLUDE_L11_BILEVEL11_HPP


#include <limits>
#include <memory>
#include <numeric>

#include "l1/l1.hpp"
#include "utils/arithmetic.hpp"

namespace proj {
namespace l11 {

inline void Bilevel(double* y, double* x, const int nrows, const int ncols,
                    const double C) {
  double *rows_max = new double[nrows];
  for (size_t i = 0; i < nrows; i++) {
    double local_sum = 0;
    for (size_t j = 0; j < ncols; j++) {
      local_sum += fabs(y[i * ncols + j]);
    }
    rows_max[i] = local_sum;
  }
  proj::l1::project(rows_max, rows_max, nrows, C);
  for (size_t i = 0; i < nrows; i++) {
    proj::l1::project(y + i * ncols, x + i * ncols, ncols, rows_max[i]);
  }
  delete[] rows_max;
}

}  // namespace l11
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L11_BILEVEL11_HPP */
