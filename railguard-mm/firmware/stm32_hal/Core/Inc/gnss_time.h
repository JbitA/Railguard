#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { uint32_t epoch; float latitude; float longitude; float speed_mps; bool valid; } gnss_fix_t;
/* Parse a valid GPRMC/GNRMC sentence and return time plus navigation fix. */
bool gnss_rmc_fix(const char *line,size_t len,gnss_fix_t *fix_out);
bool gnss_rmc_epoch(const char *line,size_t len,uint32_t *epoch_out);
