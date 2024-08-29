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

#ifndef PROJCODE_INCLUDE_L11_BILEVELPARALLEL11_HPP
#define PROJCODE_INCLUDE_L11_BILEVELPARALLEL11_HPP


#include <numeric>
#include <memory>
#include <limits>

#include "l1infty/bilevel.hpp"
#include "utils/threadpool.hpp"

namespace proj {
namespace l11 {


int BilevelParallelSub1(double* y, double* rows_max, const int nrows, const int ncols, int start, int end){
    for (size_t i = start; i < end; i++) {
        double local_sum = 0;
        for (size_t j = 0; j < ncols; j++) {
            local_sum += fabs(y[i * ncols + j]);
        }
        rows_max[i] = local_sum;
    }
    return 0;
}
int BilevelParallelSub2(double* y, double* x, double* rows_max, const int nrows, const int ncols, int start, int end){
    for (size_t i = start; i < end; i++) {
        proj::l1::project(y + i * ncols, x + i * ncols, ncols, rows_max[i]);
    }
    return 0;
}

inline void BilevelParallel(double* y, double* x, const int nrows, const int ncols,
                  const double C, int nb_workers, ThreadPool &tp) {

double *rows_max = new double[nrows];
int work_slice = nrows/nb_workers;
std::vector<std::shared_future<int>> jobs;
for (size_t w = 0; w < nb_workers; w++) {
    if (w < nb_workers-1) {
        // BilevelParallelSub1(y, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice);
        jobs.emplace_back(tp.enqueue(BilevelParallelSub1, y, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice));
    } else{ //last
        // BilevelParallelSub1(y, rows_max, nrows, ncols, w*work_slice, nrows); 
        jobs.emplace_back(tp.enqueue(BilevelParallelSub1, y, rows_max, nrows, ncols, w*work_slice, nrows));
    }
}
for (auto &&job : jobs) {
    job.get();
}
jobs.clear();


proj::l1::project(rows_max, rows_max, nrows, C);

for (size_t w = 0; w < nb_workers; w++) {
    if (w < nb_workers-1) {
        // BilevelParallelSub2(y, x, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice);
        jobs.emplace_back(tp.enqueue(BilevelParallelSub2, y, x, rows_max, nrows, ncols, w*work_slice, (w+1)*work_slice));
    } else{ //last
        // BilevelParallelSub2(y, x, rows_max, nrows, ncols, w*work_slice, nrows); 
        jobs.emplace_back(tp.enqueue(BilevelParallelSub2, y, x,  rows_max, nrows, ncols, w*work_slice, nrows));
    }
    
}

for (auto &&job : jobs) {
    job.get();
}
jobs.clear();
delete[] rows_max;
}

}  // namespace l1infty
}  // namespace proj



#endif /* PROJCODE_INCLUDE_L11_BILEVELPARALLEL11_HPP */
