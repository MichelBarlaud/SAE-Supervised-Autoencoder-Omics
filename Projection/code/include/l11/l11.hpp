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

#ifndef PROJCODE_INCLUDE_L11_L11_HPP
#define PROJCODE_INCLUDE_L11_L11_HPP

#include "l1/l1.hpp"
#include "l11/bilevel11.hpp"
#include "l11/bilevelParallel11.hpp"
#include "utils/threadpool.hpp"

namespace proj {
namespace l11 {

inline double norm(double* y, int nrows, int ncols) {
  double s = 0;
  for (size_t i = 0; i < nrows * ncols; i++) {
    s += abs(y[i]);
  }
  return s;
}

inline void project(double* y, double* x, int nrows, int ncols, const double a) {
  proj::l1::project(y, x, nrows * ncols, a);
}

inline void projectBilevel(double* y, double* x, int nrows, int ncols, const double a) {
  proj::l11::Bilevel(y, x, nrows, ncols, a);
}

inline void projectBilevelParallel(double* y, double* x, int nrows, int ncols,
                    const double C, int nb_workers) {
  ThreadPool tp(nb_workers);
  proj::l11::BilevelParallel(y, x, nrows, ncols, C, nb_workers, tp);
}

}  // namespace l11
} // namespace proj


#endif /* PROJCODE_INCLUDE_L11_L11_HPP */
