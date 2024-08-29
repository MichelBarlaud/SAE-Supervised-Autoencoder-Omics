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

#ifndef PROJCODE_INCLUDE_L1INFTY_NAIVE_HPP
#define PROJCODE_INCLUDE_L1INFTY_NAIVE_HPP


#include <numeric>
#include <vector>

#include "l1/l1.hpp"

namespace proj {
namespace l1infty {

inline void Naive(double* y, double* x, const int nrows, const int ncols,
                  const double C) {
  std::vector<double> S(nrows, 0);
  std::vector<double> a(nrows, 0);
  std::iota(std::begin(a), std::end(a), 0);

  // theta processing
  std::vector<double> SX(nrows, 0);
  std::vector<size_t> SXCard(nrows, 1);
  std::vector<double> SXOverCard(nrows, 0);
  std::vector<double> SOneOverCard(nrows, 1);

  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      auto v = y[i * ncols + j];
      S[i] += v;
      if (v > SXOverCard[i]) {
        SXOverCard[i] = v;
      }
    }
  }
  for (size_t i = 0; i < nrows; i++) {
  }
  double theta_num = std::accumulate(SXOverCard.begin(), SXOverCard.end(), 0.);
  double theta_den = nrows;
  double theta = (theta_num - C) / theta_den;

  bool changed = true;

  while (changed) {
    for (size_t a_i = 0; a_i < a.size(); a_i++) {
      size_t i = a[a_i];
      if (S[i] < theta) {
        theta_num -= SXOverCard[i];
        theta_den -= SOneOverCard[i];
        SXOverCard[i] = 0;
        SOneOverCard[i] = 0;
        a[a_i] = a.back();
        a.pop_back();
        a_i--;
        continue;
      }
      auto* yi = y + (i * ncols);
      proj::l1::project(yi, x, ncols, theta);
      SX[i] = 0;
      SXCard[i] = 0;
      theta_num -= SXOverCard[i];
      theta_den -= SOneOverCard[i];
      SXOverCard[i] = 0;
      SOneOverCard[i] = 0;
      for (size_t j = 0; j < ncols; j++) {
        if (x[j] > 0) {
          SX[i] += yi[j];
          SXCard[i]++;
        }
      }
      if (SXCard[i]) {
        SXOverCard[i] = SX[i] / SXCard[i];
        SOneOverCard[i] = 1. / SXCard[i];
        theta_num += SXOverCard[i];
        theta_den += SOneOverCard[i];
      }
    }
    double theta_prime = (theta_num - C) / theta_den;
    changed = theta_prime != theta;
    theta = theta_prime;
  }
  double* x_prime = x;
  for (size_t i = 0; i < nrows; i++) {
    if (S[i] < theta) {
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = 0;
      }
    } else {
      double mu = (SX[i] - theta) / SXCard[i];
      auto* yi = y + (i * ncols);
      auto* xi = x + (i * ncols);
      proj::l1::project(yi, xi, ncols, theta);
      for (size_t j = 0; j < ncols; j++) {
        size_t id = i * ncols + j;
        x[id] = y[id] - xi[j];
      }
    }
  }
}

}  // namespace l1infty
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTY_NAIVE_HPP */
