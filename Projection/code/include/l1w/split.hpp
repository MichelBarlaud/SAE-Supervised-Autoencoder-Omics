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

#ifndef PROJCODE_INCLUDE_L1W_SPLIT_HPP
#define PROJCODE_INCLUDE_L1W_SPLIT_HPP


#include <cstdio>
#include <iostream>

namespace proj {
namespace l1w {

void ProjWSplit(double* y, double* w, double* x, const int length,
                const double a) {
  double* aux = new double[length];
  double* aux0 = aux;
  double* waux = new double[length];
  double* waux0 = waux;
  int auxlength = 1;
  int auxlengthold = -1;
  double sumWY = *w * (*aux = *y);
  double Ws = (*waux = *w) * *w;
  double tau = (sumWY - a) / Ws;
  int i = 1;
  int iter = 1;
  for (; i < length; i++)
    if (y[i] > w[i] * tau) {
      sumWY += (aux[auxlength] = y[i]) * w[i];
      Ws += (waux[auxlength] = w[i]) * w[i];
      tau = (sumWY - a) / Ws;
      if (tau < (w[i] * y[i] - a) / (w[i] * w[i])) {
        sumWY = w[i] * y[i];
        Ws = w[i] * w[i];
        tau = (sumWY - a) / Ws;
        auxlengthold = auxlength - 1;
      }
      auxlength++;
    }
  if (auxlengthold >= 0) {
    iter++;
    auxlength -= ++auxlengthold;
    aux += auxlengthold;
    waux += auxlengthold;
    while (--auxlengthold >= 0)
      if (aux0[auxlengthold] > waux0[auxlengthold] * tau) {
        sumWY += (*(--aux) = aux0[auxlengthold]) * waux0[auxlengthold];
        Ws += (*(--waux) = waux0[auxlengthold]) * waux0[auxlengthold];
        tau = (sumWY - a) / Ws;
        auxlength++;
      }
  }
  do {
    iter++;
    auxlengthold = auxlength - 1;
    for (i = auxlength = 0; i <= auxlengthold; i++)
      if (aux[i] > waux[i] * tau) {
        aux[auxlength] = aux[i];
        waux[auxlength++] = waux[i];
      } else {
        sumWY -= aux[i] * waux[i];
        Ws -= waux[i] * waux[i];
        tau = (sumWY - a) / Ws;
      }
  } while (auxlength <= auxlengthold);

  for (i = 0; i < length; i++) x[i] = std::max(y[i] - w[i] * tau, 0.0);
  delete[] aux0;
  delete[] waux0;
}

}  // namespace l1w
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1W_SPLIT_HPP */
