'''to test the relative velocity filter function.
@author:cvhadessun
date:2021-7-15-11:14'''

from filter import RelativeVelocityFilter

kns = 1e+9
kms = 1e+3
timestamp_ns = 1

# test 1.
# rev_filter = RelativeVelocityFilter(1,1.0)
#
# res = rev_filter.Apply(1,0.5,1000.5)
#
# print(res)

# test 2.
# test same value scale diff velocity scale

filter1 = RelativeVelocityFilter(5, 45.0, 1)

filter2 = RelativeVelocityFilter(5, 0.1, 1)

value_scale = 1.0

value = 1.0
res1 = filter1.Apply(1 / kms, value_scale, value)
res2 = filter2.Apply(1 / kms, value_scale, value)

print(res1, res2)
value = 10.0
res1 = filter1.Apply(2 / kms, value_scale, value)
res2 = filter2.Apply(2 / kms, value_scale, value)

print(res1, res2)
#
value = 2.0
res1 = filter1.Apply(3 / kms, value_scale, value)
res2 = filter2.Apply(3 / kms, value_scale, value)

print(res1, res2)
