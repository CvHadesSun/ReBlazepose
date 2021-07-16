'''referring to :mediapipe/mediapipe/calculators/util/visibility_smoothing_calculator.cc'''

from .filter import LowPassFilter


class VisibilityFilter:
    def Reset(self):
        raise NotImplemented

    def Apply(self, in_landmarks, timestamp):
        raise NotImplemented


class LowPassVisibilityFilter(VisibilityFilter):
    def __init__(self,alpha):
        self.alpha_ = alpha
        self.visibility_filters_ = []

    def Reset(self):
        self.visibility_filters_ = []

    def Apply(self, in_landmarks, timestamp):
        return self._ApplyImpl(in_landmarks,timestamp)

    def _ApplyImpl(self, in_landmarks, timestamp):
        # Initializes filters for the first time or after Reset. If initialized
        self._InitializeFiltersIfEmpty(in_landmarks.shape[0])

        # Filter visibilities.
        output_landmarks = in_landmarks
        for i in range(in_landmarks.shape[0]):
            output_landmarks[i,-1] = self.visibility_filters_[i].Apply(in_landmarks[i,-1])


        return output_landmarks

    def _InitializeFiltersIfEmpty(self, num_landmark):
        if len(self.visibility_filters_) == num_landmark:
            return
        self.visibility_filters_ = [LowPassFilter(self.alpha_) for _ in range(num_landmark)]
        return
