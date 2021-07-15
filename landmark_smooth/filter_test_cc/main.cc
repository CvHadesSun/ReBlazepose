#include <algorithm>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/time/time.h"


using namespace std;

absl::Duration DurationFromNanos(int64_t nanos) {
  return absl::FromChrono(std::chrono::nanoseconds{nanos});
}

absl::Duration DurationFromMillis(int64_t millis) {
  return absl::FromChrono(std::chrono::milliseconds{millis});
}


int main()
{
    absl::Duration timestamp1 = DurationFromMillis(3);

    const int64_t new_timestamp = absl::ToInt64Nanoseconds(timestamp1);
    cout<<new_timestamp<<endl;
    return 0;
}

