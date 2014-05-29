#include <vector>
#include "algorithms.h"
#include "register.h"
#include "thrust.h"

static register_scan_algorithm<thrust_scan> register_thrust_scan;
static register_sort_algorithm<thrust_sort> register_thrust_sort;
