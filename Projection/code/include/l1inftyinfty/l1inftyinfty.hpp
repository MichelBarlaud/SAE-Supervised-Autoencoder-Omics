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

#ifndef PROJCODE_INCLUDE_L1INFTYINFTY_L1INFTYINFTY_HPP
#define PROJCODE_INCLUDE_L1INFTYINFTY_L1INFTYINFTY_HPP


#include "l1/l1.hpp"
#include "l1inftyinfty/trilevel1inftyinfty.hpp"
#include "utils/threadpool.hpp"

namespace proj {
namespace l1inftyinfty {

inline double norm(double* y, int d1, int d2, int d3) {
  double s = 0;
  for (std::size_t i = 0; i < d1; i++) {
    double max_v = 0;
    for (std::size_t j = 0; j < d2; j++) {
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        max_v = fmax(abs(y[id]), max_v);
      }
    }
    s += max_v;
  }
  return s;
}

inline void project(double* y, double* x, int d1, int d2, int d3,
                    const double a) {
  throw "unimplemented";
}

inline void projectTrilevel(double* y, double* x, int d1, int d2, int d3,
                            const double a) {
  TrilevelRec(y, x, d1, d2, d3, a);
}

inline void projectBilevelParallel(double* y, double* x, int d1, int d2, int d3,
                                   const double C, int nb_workers) {
  ThreadPool tp(nb_workers);
  TrilevelParallel(y, x, d1, d2, d3, C, nb_workers, tp);
}

}  // namespace l111
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTYINFTY_L1INFTYINFTY_HPP */
