#include <vector>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"
#include "scanbench_thrust.h"

static register_scan_algorithm<thrust_scan> register_thrust_scan;
static register_sort_algorithm<thrust_sort> register_thrust_sort;
