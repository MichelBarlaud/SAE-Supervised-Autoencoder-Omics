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

#ifndef PROJCODE_INCLUDE_UTILS_PRINT_HPP
#define PROJCODE_INCLUDE_UTILS_PRINT_HPP


#include <cstdio>
#include <iostream>
#include "utils/types.hpp"

namespace proj {

inline void PrintMatrix(double* x, const int nrows, const int ncols) {
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
        std::size_t id = i*ncols+j;
        std::cout << x[id] << ", ";
    }
    std::cout << "\n";
  }
}
inline void PrintMatrix(int* x, const int nrows, const int ncols) {
  for (std::size_t i = 0; i < nrows; i++) {
    for (std::size_t j = 0; j < ncols; j++) {
        std::size_t id = i*ncols+j;
        std::cout << x[id] << ", ";
    }
    std::cout << "\n";
  }
}

}  // namespace proj


#endif /* PROJCODE_INCLUDE_UTILS_PRINT_HPP */
