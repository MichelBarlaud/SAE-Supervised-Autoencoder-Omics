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

#pragma once

#include <numeric>
#include <memory>
#include <limits>

#include <torch/extension.h>

#include <iostream>

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

inline void projL1_(const double *y, double *x, const int length,
                    const double a) {
  double* aux = new double[length];
  double *aux0 = aux;
  int auxlength = 1;
  int auxlengthold = -1;
  double tau = (*aux = *y) - a;
  int i = 1;
  int iter = 1;
  for (; i < length; i++)
    if (y[i] > tau) {
      if ((tau += ((aux[auxlength] = y[i]) - tau) /
                  (auxlength - auxlengthold)) <= y[i] - a)
      {
        tau = y[i] - a;
        auxlengthold = auxlength - 1;
      }
      auxlength++;
    }
  if (auxlengthold >= 0) {
    iter++;
    auxlength -= ++auxlengthold;
    aux += auxlengthold;
    while (--auxlengthold >= 0)
      if (aux0[auxlengthold] > tau)
        tau += ((*(--aux) = aux0[auxlengthold]) - tau) / (++auxlength);
  }
  do {
    iter++;
    auxlengthold = auxlength - 1;
    for (i = auxlength = 0; i <= auxlengthold; i++)
      if (aux[i] > tau)
        aux[auxlength++] = aux[i];
      else
        tau += (tau - aux[i]) / (auxlengthold - i + auxlength);
  } while (auxlength <= auxlengthold);

  for (i = 0; i < length; i++)
    x[i] = (y[i] > tau ? y[i] - tau : 0.0);
  delete[] aux0;
}


inline void Bilevel(double* y, double* x, const int nrows, const int ncols,
                  const double C) {

double rows_max[nrows];
for (int i = 0; i < nrows; i++) {
    double local_max = std::numeric_limits<double>::min();
    for (int j = 0; j < ncols; j++) {
        auto v = fabs(y[i * ncols + j]);
        local_max = std::max(local_max,v);
    }
    rows_max[i] = local_max;
}
projL1_(rows_max, rows_max, nrows, C);
for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
        auto v = y[i * ncols + j];
        auto abs_v = std::fabs(v);
        x[i * ncols + j] = sgn<double>(v) * std::min(abs_v,rows_max[i]);
    }
}
}


inline torch::Tensor projL1(torch::Tensor y, const double a) {
  torch::Tensor x(y);
  projL1_(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), a);
  return x;
}

torch::Tensor l1infty_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x(y);
  Bilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("projL1", &projL1, "Projection L1 ball");
  m.def("l1infty_bilevel", &l1infty_bilevel, "l1infty bilevel");
}