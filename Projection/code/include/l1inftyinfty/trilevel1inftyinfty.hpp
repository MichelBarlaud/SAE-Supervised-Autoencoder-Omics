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

#ifndef PROJCODE_INCLUDE_L1INFTYINFTY_TRILEVEL1INFTYINFTY_HPP
#define PROJCODE_INCLUDE_L1INFTYINFTY_TRILEVEL1INFTYINFTY_HPP


#include <limits>
#include <memory>
#include <numeric>

#include "l1/l1.hpp"
#include "l1infty/l1infty.hpp"
#include "l1infty/bilevelParallel.hpp"
#include "utils/arithmetic.hpp"
#include "utils/types.hpp"
#include "utils/threadpool.hpp"

namespace proj {
namespace l1inftyinfty {

inline void TrilevelRec(double* y, double* x, int d1, int d2, int d3,
                     const double C) {
  double *d3_max = new double[d1 * d2];  // first aggregation
  for (std::size_t i = 0; i < d1; i++) {
    for (std::size_t j = 0; j < d2; j++) {
      double local_d3 = 0;
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        local_d3 = fmax(fabs(y[id]), local_d3);
      }
      d3_max[i*d2+j] = local_d3;
    }
  }
  proj::l1infty::Bilevel(d3_max, d3_max, d1, d2, C);
  for (std::size_t i = 0; i < d1; i++) {
    for (std::size_t j = 0; j < d2; j++) {
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        auto v = y[id];
        auto abs_v = std::fabs(v);
        x[id] = sgn<double>(v) * std::min(abs_v,d3_max[i*d2+j]);
      }
    }
  }
  delete[] d3_max;
}



// parallel
int TrilevelParallelSub1(double* y, double* d3_max, int d1, int d2, int d3, int start, int end){
   for (std::size_t i = start; i < end; i++) {
    for (std::size_t j = 0; j < d2; j++) {
      double local_d3 = 0;
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        local_d3 = fmax(fabs(y[id]), local_d3);
      }
      d3_max[i*d2+j] = local_d3;
    }
  }
    return 0;
}
int TrilevelParallelSub2(double* y, double* x, double* d3_max, int d1, int d2, int d3, int start, int end){
    for (std::size_t i = start; i < end; i++) {
    for (std::size_t j = 0; j < d2; j++) {
      for (std::size_t k = 0; k < d3; k++) {
        std::size_t id = COORD3D(i, j, k, d1, d2, d3);
        auto v = y[id];
        auto abs_v = std::fabs(v);
        x[id] = sgn<double>(v) * std::min(abs_v,d3_max[i*d2+j]);
      }
    }
  }
    return 0;
}

inline void TrilevelParallel(double* y, double* x, int d1, int d2, int d3,
                  const double C, int nb_workers, ThreadPool &tp) {

double *d3_max = new double[d1*d2];  // first aggregation
int work_slice = (d1)/nb_workers;

std::vector<std::shared_future<int>> jobs;
for (size_t w = 0; w < nb_workers; w++) {
    if (w < nb_workers-1) {
        // BilevelParallelSub1(y, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice);
        jobs.emplace_back(tp.enqueue(TrilevelParallelSub1, y, d3_max, d1, d2, d3, w*work_slice, (w+1)*work_slice));
    } else{ //last
        // BilevelParallelSub1(y, rows_max, nrows, ncols, w*work_slice, nrows); 
        jobs.emplace_back(tp.enqueue(TrilevelParallelSub1, y, d3_max, d1, d2, d3, w*work_slice, d1));
    }
}
for (auto &&job : jobs) {
    job.get();
}
jobs.clear();


proj::l1infty::BilevelParallel(d3_max, d3_max, d1, d2, C, nb_workers, tp);

for (size_t w = 0; w < nb_workers; w++) {
    if (w < nb_workers-1) {
        // BilevelParallelSub2(y, x, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice);
        jobs.emplace_back(tp.enqueue(TrilevelParallelSub2, y, x, d3_max, d1, d2, d3, w*work_slice, (w+1)*work_slice));
    } else{ //last
        // BilevelParallelSub2(y, x, rows_max, nrows, ncols, w*work_slice, nrows); 
        jobs.emplace_back(tp.enqueue(TrilevelParallelSub2, y, x, d3_max, d1, d2, d3, w*work_slice, d1));
    }
    
}

for (auto &&job : jobs) {
    job.get();
}
jobs.clear();
delete[] d3_max;
}


}  // namespace l111
}  // namespace proj


#endif /* PROJCODE_INCLUDE_L1INFTYINFTY_TRILEVEL1INFTYINFTY_HPP */
