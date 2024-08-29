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

#ifndef PROJCODE_INCLUDE_L1W_SORT_HPP
#define PROJCODE_INCLUDE_L1W_SORT_HPP


#include <cstdio>
#include <iostream>

#include "utils/Sort.hpp"


namespace proj {
namespace l1w {

void ProjWSort(const double* y, double* w, double* x, const int length,
               const double a) {
  double* z = (double*)malloc(length * sizeof(double));
  int* z_perm = (int*)malloc(length * sizeof(int));

  int i;
  for (i = 0; i < length; ++i) {
    z[i] = y[i] / w[i];
    z_perm[i] = i;
  }
  quicksort(z, z_perm, 0, length - 1);

  i = 0;
  double sumWY = w[z_perm[i]] * y[z_perm[i]];
  double Ws = w[z_perm[i]] * w[z_perm[i]];
  double tau = (sumWY - a) / Ws;
  i++;
  while ((i < length) && (z[i] > tau)) {
    sumWY += w[z_perm[i]] * y[z_perm[i]];
    Ws += w[z_perm[i]] * w[z_perm[i]];
    tau = (sumWY - a) / Ws;
    i++;
  }

  // printf("%f\n",tau);

  for (i = 0; i < length; i++)
    x[i] = (y[i] > w[i] * tau ? y[i] - w[i] * tau : 0.0);
  free(z);
  free(z_perm);
}

}  // namespace l1w
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1W_SORT_HPP */
