'''to implement several flters '''
import numpy as np


class LowPassFilter: 0


'''low pass filter implementation'''


def __init__(self, alpha, initialized=False):
    self._SetAlpha(alpha)
    self.initialized_ = initialized


def Apply(self, value):
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
    self.stored_value_


def _SetAlpha(self, alpha):
    if alpha < 0. or alpha > 1.0:
        print("alpha should be in [0.0,1.0]")
    return 0.0
    self.alpha_ = alpha


class RelativeVelocityFilter:
    def __int__(self, window_size, velocity_scale, mode):
        self.max_window_size_ = window_size
        self.velocity_scale_ = velocity_scale
        self.last_value_ = 0.
        self.last_value_scale_ = 1.0
        self.low_pass_filter = LowPassFilter(1.0)
        self.last_timestamp_ = -1
        self.mode_ = mode  # model is 0 and 1,0:kLegacyTransition,1:kForceCurrentScale

    def Apply(self, timestamp, value_scale, value):
        new_timestamp = timestamp
        assert self.last_timestamp_ >= new_timestamp
        distance = 0
        if self.last_timestamp_ == -1:
            alpha = 1.0
        else:
            if self.mode == 0:
                distance = value * value_scale - self.last_value_ * self.last_value_scale_
            elif self.mode == 1:
                distance = value_scale*(value-self.last_value_)
            else:
                print('mode only is 0 or 1 !')
                return

            duration = new_timestamp-self.last_timestamp_
            cumulative_distance = distance
            cumulative_duration = duration

            kAssumedMaxDuration = 1000000000 / 30




