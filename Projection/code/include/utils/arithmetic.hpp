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

#ifndef PROJCODE_INCLUDE_UTILS_ARITHMETIC_HPP
#define PROJCODE_INCLUDE_UTILS_ARITHMETIC_HPP

#include <limits>
#include <random>

namespace proj {


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

}  // namespace proj


#endif /* PROJCODE_INCLUDE_UTILS_ARITHMETIC_HPP */
