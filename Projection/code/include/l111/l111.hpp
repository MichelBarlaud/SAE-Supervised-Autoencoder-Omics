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

#ifndef PROJCODE_INCLUDE_L111_L111_HPP
#define PROJCODE_INCLUDE_L111_L111_HPP

#include "l1/l1.hpp"
#include "l111/trilevel111.hpp"
#include "utils/threadpool.hpp"

namespace proj {
namespace l111 {

inline double norm(double* y, int d1, int d2, int d3) {
  double s = 0;
  for (size_t i = 0; i < d1 * d2 * d3; i++) {
    s += abs(y[i]);
  }
  return s;
}

inline void project(double* y, double* x, int d1, int d2, int d3,
                    const double a) {
  proj::l1::project(y, x, d1 * d2 * d3, a);
}

inline void projectTrilevel(double* y, double* x, int d1, int d2, int d3,
                            const double a) {
  Trilevel(y, x, d1, d2, d3, a);
}

inline void projectBilevelParallel(double* y, double* x, int d1, int d2, int d3,
                                   const double C, int nb_workers) {
  ThreadPool tp(nb_workers);
  throw "unimplemented";
}

}  // namespace l111
}  // namespace proj

#endif /* PROJCODE_INCLUDE_L111_L111_HPP */
