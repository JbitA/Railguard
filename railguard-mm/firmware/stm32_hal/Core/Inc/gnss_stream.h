#pragma once
#include <stddef.h>
#include <stdint.h>

#define RG_GNSS_LINE_MAX 160u

typedef void (*rg_gnss_line_fn)(const char *line, size_t len, void *context);

typedef struct {
    char line[RG_GNSS_LINE_MAX];
    size_t len;
    uint32_t lines;
    uint32_t overflows;
} rg_gnss_stream_t;

void rg_gnss_stream_init(rg_gnss_stream_t *stream);
void rg_gnss_stream_feed(
    rg_gnss_stream_t *stream, const uint8_t *data, size_t len,
    rg_gnss_line_fn callback, void *context);
