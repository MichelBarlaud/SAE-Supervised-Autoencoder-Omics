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


#include <numeric>
#include <memory>
#include <limits>
#include <torch/extension.h>
#include <iostream>

#include "l1/l1.hpp"
#include "l1w/l1w.hpp"
#include "l2/l2.hpp"
#include "l11/l11.hpp"
#include "l12/l12.hpp"
#include "l1infty/l1infty.hpp"
#include "l1inftyinfty/l1inftyinfty.hpp"
#include "l111/l111.hpp"


// 1 D
inline torch::Tensor proj_l1(torch::Tensor y, const double a) {
  torch::Tensor x = y.detach().clone();
  proj::l1::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), a);
  return x;
}

inline torch::Tensor proj_l1w(torch::Tensor y, torch::Tensor w, const double a) {
  torch::Tensor x = y.detach().clone();
  proj::l1w::project(y.data_ptr<double>(), w.data_ptr<double>(), x.data_ptr<double>(), y.size(0), a);
  return x;
}

inline torch::Tensor proj_l2(torch::Tensor y, const double a) {
  torch::Tensor x = y.detach().clone();
  proj::l2::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), a);
  return x;
}


// 2D
torch::Tensor proj_l1infty(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l1infty::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

torch::Tensor proj_l1infty_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l1infty::projectBilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

torch::Tensor proj_l1infty_bilevel_parallel(torch::Tensor y, const double C, int nb_workers) {
  torch::Tensor x = y.detach().clone();
  proj::l1infty::projectBilevelParallel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C, nb_workers);
  return x;
}

torch::Tensor proj_l11(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l11::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

torch::Tensor proj_l11_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l11::projectBilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

torch::Tensor proj_l11_bilevel_parallel(torch::Tensor y, const double C, int nb_workers) {
  torch::Tensor x = y.detach().clone();
  proj::l11::projectBilevelParallel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C, nb_workers);
  return x;
}

torch::Tensor proj_l12(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l12::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

torch::Tensor proj_l12_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l12::projectBilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), C);
  return x;
}

// 3D
torch::Tensor proj_l111(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l111::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), y.size(2), C);
  return x;
}

torch::Tensor proj_l111_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l111::projectTrilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), y.size(2), C);
  return x;
}

torch::Tensor proj_l1inftyinfty(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l1inftyinfty::project(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), y.size(2), C);
  return x;
}

torch::Tensor proj_l1inftyinfty_bilevel(torch::Tensor y, const double C) {
  torch::Tensor x = y.detach().clone();
  proj::l1inftyinfty::projectTrilevel(y.data_ptr<double>(), x.data_ptr<double>(), y.size(0), y.size(1), y.size(2), C);
  return x;
}


inline double norm_l1(torch::Tensor y) {
  return proj::l1::norm(y.data_ptr<double>(), y.size(0));
}

inline double norm_l1w(torch::Tensor y, torch::Tensor w) {
  return proj::l1w::norm(y.data_ptr<double>(), w.data_ptr<double>(), y.size(0));
}
inline double norm_l2(torch::Tensor y) {
  return proj::l2::norm(y.data_ptr<double>(), y.size(0));
}
inline double norm_l11(torch::Tensor y) {
  return proj::l11::norm(y.data_ptr<double>(), y.size(0), y.size(1));
}
inline double norm_l12(torch::Tensor y) {
  return proj::l12::norm(y.data_ptr<double>(), y.size(0), y.size(1));
}
inline double norm_l1infty(torch::Tensor y) {
  return proj::l1infty::norm(y.data_ptr<double>(), y.size(0), y.size(1));
}
inline double norm_l111(torch::Tensor y) {
  return proj::l111::norm(y.data_ptr<double>(), y.size(0), y.size(1), y.size(2));
}
inline double norm_l1inftyinfty(torch::Tensor y) {
  return proj::l1inftyinfty::norm(y.data_ptr<double>(), y.size(0), y.size(1), y.size(2));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Projection methods for numpy binding";
  m.def("proj_l1", &proj_l1, "Projection L1 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l1w", &proj_l1w, "Projection weighted L1 ball", py::arg("y"), py::arg("w"), py::arg("radius")=1.);
  m.def("proj_l2", &proj_l2, "Projection L2 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l1infty", &proj_l1infty, "Projection L,1,infinity ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l1infty_bilevel", &proj_l1infty_bilevel, "Bilevel Projection L,1,infinity ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l11", &proj_l11, "Projection L11 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l11_bilevel", &proj_l11_bilevel, "Bilevel Projection L11 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l12", &proj_l12, "Projection L12 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l12_bilevel", &proj_l12_bilevel, "Bilevel Projection L12 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("norm_l1", &norm_l1, "Norm L1", py::arg("y"));
  m.def("norm_l1w", &norm_l1w, "Norm weighted L1", py::arg("y"), py::arg("w"));
  m.def("norm_l2", &norm_l2, "Norm L2", py::arg("y"));
  m.def("norm_l11", &norm_l11, "Norm L11", py::arg("y"));
  m.def("norm_l12", &norm_l12, "Norm L12", py::arg("y"));
  m.def("norm_l1infty", &norm_l1infty, "Norm L,1,infinity", py::arg("y"));
  m.def("proj_l1infty_bilevel_parallel", &proj_l1infty_bilevel_parallel, "Bilevel Projection L,1,infinity ball", py::arg("y"), py::arg("radius")=1., py::arg("nb_workers")=1);
  m.def("proj_l11_bilevel_parallel", &proj_l11_bilevel_parallel, "Bilevel Projection L11 ball", py::arg("y"), py::arg("radius")=1., py::arg("nb_workers")=1);
  m.def("norm_l111", &norm_l111, "Norm L111", py::arg("y"));
  m.def("norm_l1inftyinfty", &norm_l1inftyinfty, "Norm L,1,infinity,infinity", py::arg("y"));
  m.def("proj_l111", &proj_l111, "Projection L111 ball", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l111_bilevel", &proj_l111_bilevel, "Projection L111 bilevel", py::arg("y"), py::arg("radius")=1.);
  m.def("proj_l1inftyinfty_bilevel", &proj_l1inftyinfty_bilevel, "Projection L,1,infinity,infinity bilevel", py::arg("y"), py::arg("radius")=1.);
}