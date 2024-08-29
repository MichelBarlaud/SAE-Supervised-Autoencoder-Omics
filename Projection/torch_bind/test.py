import torch
import projections



eta = 1
# 1D
# Test L1
y = torch.rand(1000).double()
x = projections.proj_l1(y, eta)
assert abs(projections.norm_l1(x) - eta) <= 1e-4, "L1 proj fail"

# Test L1W
y = torch.rand(1000).double()
w = torch.abs(torch.rand(1000)).double()
x = projections.proj_l1w(y, w, eta)
assert abs(projections.norm_l1w(x, w)-eta) <= 1e-4, "L1W proj fail"
assert abs(torch.dot(x, w)-eta) <= 1e-4, "L1W proj fail"

# Test L2
y = torch.rand(1000).double()
y = torch.add(y, 1)
x = projections.proj_l2(y, eta)
assert abs(projections.norm_l2(x)-eta) <= 1e-4, "L2 proj fail"
assert abs(torch.dot(x, x)-eta) <= 1e-4, "L2 proj fail"

eta = 5

# 2D
# Test L11
y = torch.rand(1000,1000).double()
x = projections.proj_l11(y, eta)
assert abs(projections.norm_l11(x)-eta) <= 1e-4, "L11 proj fail"
assert abs(torch.sum(x)-eta) <= 1e-4, "L11 proj fail"

# Test L11 bilevel
y = torch.rand(1000,1000).double()
x = projections.proj_l11_bilevel_parallel(y, eta, 12)
assert abs(projections.norm_l11(x)-eta) <= 1e-4, "L11 bilevel proj fail"
assert abs(torch.sum(x)-eta) <= 1e-4, "L11 bilevel proj fail"


y = torch.rand(1000,1000).double()
# Test L1infty
x = projections.proj_l1infty(y, eta)
assert abs(projections.norm_l1infty(x)-eta) <= 1e-4, "L1infty proj fail"
assert abs(torch.sum(torch.max(x,1).values)-eta) <= 1e-4, "L1infty proj fail"


# Test L1infty bilevel
y = torch.rand(1000,1000).double()
x = projections.proj_l1infty_bilevel_parallel(y, eta, 12)
assert abs(projections.norm_l1infty(x)-eta) <= 1e-4, "L1infty bilevel proj fail"
assert abs(torch.sum(torch.max(x,1).values)-eta) <= 1e-4, "L1infty bilevel proj fail"

y = torch.rand(1000,1000).double()
# Test L12
x = projections.proj_l12(y, eta)
assert abs(projections.norm_l12(x)-eta) <= 1e-4, "L12 proj fail"

# Test L12 bilevel
y = torch.rand(1000,1000).double()
x = projections.proj_l12_bilevel(y,radius=eta)
assert abs(projections.norm_l12(x)-eta) <= 1e-4, "L12 bilevel proj fail"


# 3D

eta = 5
# Test L111
y = torch.rand(1000,100,10).double()
x = projections.proj_l111(y, eta)
assert abs(projections.norm_l111(x)-eta) <= 1e-4, "L111 proj fail"

# Test L111 tri-level
y = torch.rand(1000,100,10).double()
x = projections.proj_l111_bilevel(y, eta)
assert abs(projections.norm_l111(x)-eta) <= 1e-4, "L111 trilevel fail"

# Test L1inftyinfty tri-level
y = torch.rand(1000,100,10).double()
x = projections.proj_l1inftyinfty_bilevel(y, eta)
assert abs(projections.norm_l1inftyinfty(x)-eta) <= 1e-4, "L1inftyinfty trilevel fail"
