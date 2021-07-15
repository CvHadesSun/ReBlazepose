'''to test the low pass filter function.
@author:cvhadessun
date:2021-7-15-10:51'''

from filter import LowPassFilter

low_pass_filter = LowPassFilter(0.5)

result1 = low_pass_filter.Apply(2.0)

result = low_pass_filter.Apply(100.0)

print(result1,result)