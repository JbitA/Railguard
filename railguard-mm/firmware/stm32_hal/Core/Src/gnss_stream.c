#include "gnss_stream.h"
#include <string.h>

void rg_gnss_stream_init(rg_gnss_stream_t *stream) {
    if (stream == NULL) return;
    memset(stream, 0, sizeof(*stream));
}

void rg_gnss_stream_feed(
    rg_gnss_stream_t *stream, const uint8_t *data, size_t len,
    rg_gnss_line_fn callback, void *context) {
    if (stream == NULL || data == NULL) return;

    for (size_t i = 0; i < len; ++i) {
        const char c = (char)data[i];
        if (c == '$') {
            stream->len = 0u;
            stream->line[stream->len++] = c;
            continue;
        }
        if (stream->len == 0u) continue;

        if (c == '\r' || c == '\n') {
            if (stream->len > 1u && callback != NULL) {
                callback(stream->line, stream->len, context);
                stream->lines++;
            }
            stream->len = 0u;
            continue;
        }

        if (stream->len + 1u >= RG_GNSS_LINE_MAX) {
            stream->len = 0u;
            stream->overflows++;
            continue;
        }
        stream->line[stream->len++] = c;
    }
}
