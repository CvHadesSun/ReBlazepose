'''to test the low pass filter function.
@author:cvhadessun
date:2021-7-15-10:51'''

from filter import LowPassFilter as filter1
from filter2 import LowPassFilter as filter2

f1 = filter1(0.5)
f2 = filter2(0.5)

print(f1.ApplyWithAlpha(4,0.5),f2(4))
print(f1.Apply(5),f2(5))
print(f1.Apply(6),f2(6))
print(f1.Apply(7),f2(7))
