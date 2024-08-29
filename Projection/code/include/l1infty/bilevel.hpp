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

#ifndef PROJCODE_INCLUDE_L1INFTY_BILEVEL_HPP
#define PROJCODE_INCLUDE_L1INFTY_BILEVEL_HPP


#include <numeric>
#include <memory>
#include <limits>

#include "l1/l1.hpp"

namespace proj {
namespace l1infty {

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

inline void Bilevel(double* y, double* x, const int nrows, const int ncols,
                  const double C) {

double *rows_max = new double[nrows];
for (size_t i = 0; i < nrows; i++) {
    double local_max = std::numeric_limits<double>::min();
    for (size_t j = 0; j < ncols; j++) {
        auto v = fabs(y[i * ncols + j]);
        local_max = std::max(local_max,v);
    }
    rows_max[i] = local_max;
}
proj::l1::project(rows_max, rows_max, nrows, C);
for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
        auto v = y[i * ncols + j];
        auto abs_v = std::fabs(v);
        x[i * ncols + j] = sgn<double>(v) * std::min(abs_v,rows_max[i]);
    }
}
delete[] rows_max;
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_BILEVEL_HPP */
