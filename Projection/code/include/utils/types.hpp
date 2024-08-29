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

#ifndef PROJCODE_INCLUDE_UTILS_TYPES_HPP
#define PROJCODE_INCLUDE_UTILS_TYPES_HPP

/**
 * @brief This macro is used to linearize the coordinate of 
 * a matrix encoded into a linear structure such as a double*
 * 
 */
#define COORD2D(i, j, nrows, ncols) (i * ncols + j)

/**
 * @brief This macro is used to linearize the coordinate of 
 * a 3 dimensionnal tensor encoded into a linear structure such as a double*
 * 
 */
#define COORD3D(i, j, k, d1, d2, d3) (i * (d2*d3) + j *d3 + k)


namespace proj {

#define DATASIZE 8

union DtB {
  double val;
  unsigned char byte[sizeof(double)];
};
typedef union DtB Dtb;


struct ValueCoord {
  double value;
  long int coordinates;
};

bool operator<(ValueCoord const & a, ValueCoord const & b)
{
    return a.value < b.value;
}

bool operator>(ValueCoord const & a, ValueCoord const & b)
{
    return a.value > b.value;
}

#define KahanSum(s, v, c, t, y) \
  y = v - c;                    \
  t = s + y;                    \
  c = (t - s) - y;              \
  s = t;
#define KahanSumDel(s, v, c, t, y) \
  y = v + c;                       \
  t = s - y;                       \
  c = (t - s) + y;                 \
  s = t;

}  // namespace proj


#endif /* PROJCODE_INCLUDE_UTILS_TYPES_HPP */
