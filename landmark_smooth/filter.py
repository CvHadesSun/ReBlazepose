'''to implement several flters '''
import numpy as np
import math
from decimal import *

#
class LowPassFilter:
    '''referring to:mediapipe/mediapipe/util/filtering/low_pass_filter.cc(.h)
    '''

    def __init__(self, alpha, initialized=False):
        self.alpha_ = alpha
        self.initialized_ = initialized

    def Apply(self, value):
        result = 0.0
        if self.initialized_:
            result = self.alpha_ * value + (1.0 - self.alpha_) * self.stored_value_
        else:
            result = value
            self.initialized_ = True
        self.raw_value_ = value
        self.stored_value_ = result

        return result

    def ApplyWithAlpha(self, value, alpha):
        self._SetAlpha(alpha)
        return self.Apply(value)

    def HasLastRawValue(self):
        return self.initialized_

    def LastRawValue(self):
        return self.raw_value_

    def LastValue(self):
        return self.stored_value_

    def _SetAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.alpha_ = alpha




class RelativeVelocityFilter:
    '''referring to :mediapipe/mediapipe/util/filtering/relative_velocity_filter.cc(.h)'''

    def __init__(self, window_size, velocity_scale, mode=0):
        self.max_window_size_ = window_size
        self.velocity_scale_ = velocity_scale
        self.last_value_ = 0.
        self.last_value_scale_ = 1.0
        self.low_pass_filter = LowPassFilter(1.0)
        self.last_timestamp_ = -1
        self.mode_ = mode  # model is 0 and 1,0:kLegacyTransition,1:kForceCurrentScale
        self.window_ = []

    def Apply(self, timestamp, value_scale, value):
        kNanoSecondsToSecond = 1e-9
        new_timestamp = timestamp / kNanoSecondsToSecond  # tranform int time to ns.
        # print(new_timestamp)
        assert self.last_timestamp_ <= new_timestamp
        distance = 0

        kAssumedMaxDuration = 1000000000 / 30  # 1/30 per frame   unit:ns

        alpha = 1.0
        if self.last_timestamp_ == -1:
            alpha = 1.0
        else:
            if self.mode_ == 0:
                distance = value * value_scale - self.last_value_ * self.last_value_scale_
            elif self.mode_ == 1:
                distance = value_scale * (value - self.last_value_)
            else:
                print('mode only is 0 or 1 !')
                return 0.0

            # debug
            # print(distance)
            duration = new_timestamp - self.last_timestamp_
            # print(duration)
            cumulative_distance = distance
            cumulative_duration = duration
            max_cumulative_duration = (1 + len(self.window_)) * kAssumedMaxDuration
            if len(self.window_) > 0:
                for i in range(len(self.window_) - 1, -1, -1):
                    if cumulative_duration + self.window_[i][1] >= max_cumulative_duration:
                        break
                    cumulative_distance += self.window_[i][0]
                    cumulative_duration += self.window_[i][1]

            velocity = cumulative_distance / (cumulative_duration * kNanoSecondsToSecond)
            alpha = 1.0 - 1.0 / (1.0 + self.velocity_scale_ * abs(velocity))
            self.window_.append([distance, duration])

            if len(self.window_) > self.max_window_size_:
                del self.window_[0]
        self.last_value_ = value
        self.last_value_scale_ = value_scale
        self.last_timestamp_ = new_timestamp
        # print(value,alpha)
        return self.low_pass_filter.ApplyWithAlpha(value, alpha)


class OneEuroFilter:
    '''referring to:mediapipe/mediapipe/util/filtering/one_euro_filter.cc(.h)'''

    def __init__(self, frequency, min_cutoff, beta, derivate_cutoff):
        self.frequency_ = float(frequency)
        self.min_cutoff_ = float(min_cutoff)
        self.beta_ = float(beta)
        self.derivate_cutoff_ = float(derivate_cutoff)
        self.last_time_ = 0
        self.x_ = LowPassFilter(self._GetAlpha(self.min_cutoff_))
        self.dx_ = LowPassFilter(self._GetAlpha(self.derivate_cutoff_))

    def Apply(self, timestamp, value_scale, value):

        '''input: timestamp unit is : s'''

        # print('input:',value)
        kNanoSecondsToSecond = 1e-9  # ns tranform
        new_timestamp = timestamp  # s-->ns

        # if self.last_time_ >= new_timestamp:
        #     print("New timestamp is equal or less than the last one.")
        #     return value

        if self.last_time_  and new_timestamp:
            self.frequency_ = 1.0 / (new_timestamp - self.last_time_)
        self.last_time_ = new_timestamp
        if self.x_.HasLastRawValue():
            # print(self.x_.LastValue())
            dvalue = (value - self.x_.LastRawValue())  * value_scale*self.frequency_
        else:
            dvalue = 0.0
        # print("befor:",dvalue,self._GetAlpha(self.derivate_cutoff_))
        edvalue = self.dx_.ApplyWithAlpha(dvalue, self._GetAlpha(self.derivate_cutoff_))
        # print("after:",edvalue)
        # use it to update cutoff frequency
        cutoff = self.min_cutoff_ + self.beta_ * math.fabs(edvalue)

        # filter the value
        return self.x_.ApplyWithAlpha(value, self._GetAlpha(cutoff))

    def _GetAlpha(self, cutoff):
        te = 1.0 / self.frequency_
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _SetFrequency(self, frequency):
        assert frequency > 0.0, "frequency should be > 0"
        self.frequency_ = frequency

    def _SetMinCutoff(self, min_cutoff):
        assert min_cutoff > 0, "min_cutoff should be > 0"
        self.min_cutoff_ = min_cutoff

    def _SetBeta(self, beta):
        self.beta_ = beta

    def _SetDerivateCutoff(self, derivate_cutoff):
        assert derivate_cutoff > 0, "derivate_cutoff should be > 0"
        self.derivate_cutoff_ = derivate_cutoff

# a=[1,2,3,4]
# # print(a)
# for i in range(len(a)-1,-1,-1):
#     print(a[i])
