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

#ifndef PROJCODE_INCLUDE_L2_L2_HPP
#define PROJCODE_INCLUDE_L2_L2_HPP

#include <cstdio>
#include <cmath>

namespace proj {
namespace l2 {

inline double norm(double* y, int dimension) {
  double s = 0;
  for (std::size_t i = 0; i < dimension; i++) {
    s += y[i] * y[i];
  }
  return sqrt(s);
}

inline void project(double* y, double* x, int dimension, const double a) {
  double n = norm(y, dimension);
  double factor = a / n;
  if (n <= a) {
    factor = 1.;
  }
  for (std::size_t i = 0; i < dimension; i++) {
    x[i] = y[i] * factor;
  }
}

}  // namespace l2
}  // namespace proj

#endif /* PROJCODE_INCLUDE_L2_L2_HPP */
