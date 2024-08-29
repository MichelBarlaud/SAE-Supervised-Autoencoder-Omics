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

#ifndef PROJCODE_INCLUDE_L1_L1_HPP
#define PROJCODE_INCLUDE_L1_L1_HPP

#include "l1/bucket.hpp"
#include "l1/bucket_filter.hpp"
#include "l1/condat.hpp"

namespace proj {
namespace l1 {

inline double norm(double* y, int dimension) {
  double s = 0;
  for (size_t i = 0; i < dimension; i++)  {
    s += fabs(y[i]);
  }
  return s;
}

inline void project(double* y, double* x, int dimension, const double a) {
  ProjC(y, x, dimension, a);
}


}  // namespace l1
} // namespace proj


#endif /* PROJCODE_INCLUDE_L1_L1_HPP */
