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

#ifndef PROJCODE_INCLUDE_L12_L12_HPP
#define PROJCODE_INCLUDE_L12_L12_HPP

#include "l1/l1.hpp"
#include "l2/l2.hpp"

namespace proj {
namespace l12 {

inline double norm(double* y, int nrows, int ncols) {
  double s = 0;
  for (size_t i = 0; i < nrows ; i++) {
    s += proj::l2::norm(y+ i* ncols, ncols);
  }
  return s;
}

inline void projectBilevel(double* y, double* x, int nrows, int ncols, const double a) {
  double *rows_max = new double[nrows];
  for (size_t i = 0; i < nrows; i++) {
    double local_sum = 0;
    for (size_t j = 0; j < ncols; j++) {
      auto v =  fabs(y[i * ncols + j]);
      local_sum += v*v;
    } 
    rows_max[i] = local_sum;
  }
  proj::l1::project(rows_max, rows_max, nrows, a);
  for (size_t i = 0; i < nrows; i++) {
    proj::l2::project(y + i * ncols, x + i * ncols, ncols, rows_max[i]);
  }
  delete[] rows_max;
}

inline void project(double* y, double* x, int nrows, int ncols, const double a) {
  projectBilevel(y, x, nrows, ncols, a);
}


}  // namespace l11
} // namespace proj


#endif /* PROJCODE_INCLUDE_L12_L12_HPP */
