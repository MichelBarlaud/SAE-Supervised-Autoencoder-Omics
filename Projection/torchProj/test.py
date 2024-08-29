import torch
import l1inftyB_cpp


y = torch.rand(10)

print(y)
x = l1inftyB_cpp.projL1(y.double(),1)
print(x)
print(sum(x))


y = torch.rand(100,100)
x = l1inftyB_cpp.l1infty_bilevel(y.double(),5)
s = 0
for i in range(100):
    s += max(x[i])
print(s)