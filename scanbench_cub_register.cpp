#include "scanbench_register.h"
#include "scanbench_cub.h"

static register_scan_algorithm<cub_scan> register_cub_scan;
static register_sort_algorithm<cub_sort> register_cub_sort;
