#include <algorithm>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/time/time.h"


using namespace std;
using namespace absl;
absl::Duration DurationFromNanos(int64_t nanos) {
  return absl::FromChrono(std::chrono::nanoseconds{nanos});
}

absl::Duration DurationFromMillis(int64_t millis) {
  return absl::FromChrono(std::chrono::milliseconds{millis});
}


int main()
{
    absl::Duration timestamp1 = DurationFromMillis(3);
    
    const int64_t new_timestamp = absl::ToInt64Nanoseconds(timestamp1); //3000000 ns
    // int64_t new_timestamp = 2;  //us?
    Duration timestamp_us = Microseconds(new_timestamp); //3000000 * 1e-6
    cout<<new_timestamp<<endl;  //
    cout<<timestamp_us<<endl;  //3000000 us * 1e -6 = 3s
    const int64_t new_timestamp_ns = absl::ToInt64Nanoseconds(timestamp_us); //3s --> 3e+9 ns
    cout<<new_timestamp_ns<<endl;
    return 0;
}

