/* Fast BinaryFILE scanner. stdin: decompressed ITCH; stdout: one-symbol ITCH.
 * Counts go to stderr as JSON. No order reconstruction or model logic here.
 * Build: cc -O3 -Wall -Wextra -Werror scripts/itch_filter.c -o build/itch-filter
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "ITCH filter: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 2 || strlen(argv[1]) < 1 || strlen(argv[1]) > 8)
        fail("expected an ASCII symbol of 1..8 bytes");
    char symbol[8];
    memset(symbol, ' ', sizeof symbol);
    memcpy(symbol, argv[1], strlen(argv[1]));
    unsigned char sizes[256] = {0};
    const char *types = "SRHYLVWKJhAFECXDUPQBIN";
    const unsigned char lengths[] = {12,39,25,20,26,35,12,28,35,21,36,40,31,36,23,19,35,44,40,19,50,20};
    for (size_t i = 0; i < strlen(types); ++i) sizes[(unsigned char)types[i]] = lengths[i];
    unsigned char header[2], body[100];
    int locate = -1, terminated = 0;
    unsigned long long count = 0, retained = 0;
    size_t event_count = 0;
    const char *events = "OSQMEC";
    size_t read_count;
    while ((read_count = fread(header, 1, 2, stdin)) != 0) {
        if (read_count != 2 || terminated) fail("truncated header or data after terminator");
        unsigned int length = ((unsigned int)header[0] << 8) | header[1];
        if (length == 0) {
            terminated = 1;
            if (fwrite(header, 1, 2, stdout) != 2) fail("output write failed");
            continue;
        }
        if (length < 11 || length > sizeof body) fail("invalid frame length");
        if (fread(body, 1, length, stdin) != length) fail("truncated payload");
        if (sizes[body[0]] != length) fail("unknown type or incorrect message length");
        int current = ((int)body[1] << 8) | body[2];
        if (count == 0 && (body[0] != 'S' || body[11] != 'O')) fail("missing start event");
        if (body[0] == 'S') {
            if (event_count >= 6 || body[11] != events[event_count++]) fail("invalid daily event sequence");
        }
        if (body[0] == 'R' && memcmp(body + 11, symbol, 8) == 0) {
            if (locate >= 0) fail("duplicate symbol directory");
            locate = current;
        }
        if (body[0] == 'S' || (locate >= 0 && current == locate)) {
            if (fwrite(header, 1, 2, stdout) != 2 || fwrite(body, 1, length, stdout) != length)
                fail("output write failed");
            ++retained;
        }
        ++count;
    }
    if (ferror(stdin) || fflush(stdout) != 0) fail("stream IO failed");
    if (locate < 0 || event_count != 6) fail("missing symbol or incomplete daily events");
    fprintf(stderr, "{\"all_messages\":%llu,\"retained_messages\":%llu,"
                    "\"system_events\":[\"O\",\"S\",\"Q\",\"M\",\"E\",\"C\"],"
                    "\"binaryfile_terminator_seen\":%s}\n",
                    count, retained, terminated ? "true" : "false");
    return 0;
}
