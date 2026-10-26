# TailTiny (vendored)

`tailtiny/` and `wireguard_lwip/` are [TailTiny](https://github.com/Asimmimx/tailtiny) 3.0.2 at commit `450ad519810f0a7560306e1389c65acf2c48828c`: `components/tailtiny/` there, with the `wireguard_lwip` it nests moved up beside it so ESP-IDF finds both. Line endings were converted from CRLF to LF. Licenses: [TailTiny-MIT](../../../LICENSES/TailTiny-MIT.txt), [wireguard-lwip-BSD-3-Clause](../../../LICENSES/wireguard-lwip-BSD-3-Clause.txt), [STROBE-x25519-MIT](../../../LICENSES/STROBE-x25519-MIT.txt). The firmware uses them through `main/port_tailnet.cpp` (docs/tailnet.md).

`tailtiny-hermes.patch` is the complete diff from upstream, after the line-ending conversion. To update: copy the new upstream `components/tailtiny` the same way, then reapply that patch and resolve what no longer applies.

## Changes

Build:

- Both `CMakeLists.txt` compile their sources only with `CONFIG_HG_TAILNET`; their requirements stay unconditional, because ESP-IDF resolves requirements before the sdkconfig exists. TailTiny's Kconfig menu depends on `HG_TAILNET`.
- `idf_component.yml` drops the nested `wireguard_lwip` path and adds `espressif/cjson`: ESP-IDF 6 no longer ships cJSON as `json`. `REQUIRES` names `espressif__cjson` and `esp_driver_uart` (ESP-IDF 6 dropped UART from `driver`).
- `src/ml_cellular.c` (the 4G modem driver) isn't compiled: cellular stays off, and the file includes `driver/uart.h` outside its `#ifdef`.
- `-Wno-unterminated-string-initialization` for both components: GCC 15 flags byte arrays initialised from strings of exactly their length (HSalsa20's sigma, WireGuard's protocol labels), which is deliberate.

ESP-IDF 6 / Mbed TLS 4:

- `src/ml_noise.c`: the Noise control channel's ChaCha20-Poly1305 uses wireguard-lwip's reference implementation instead of Mbed TLS's, now private. Tailscale's nonce is 4 zero bytes and a big-endian counter, so the counter is passed byte-swapped; checked against Python's `cryptography` for several counters.
- `src/ml_derp.c`, `include/tailtiny_internal.h`: the relay (DERP) TLS connection uses PSA randomness on Mbed TLS 4; the entropy and CTR-DRBG contexts remain only for Mbed TLS 3.

Behaviour:

- `src/ml_derp.c`: the relay's certificate is checked against the firmware's CA bundle (`MBEDTLS_SSL_VERIFY_REQUIRED`, `esp_crt_bundle_attach`). Upstream didn't verify it. WireGuard already encrypts what the relay carries; this stops a network in the middle from posing as the relay.
- `src/ml_coord.c`, `include/tailtiny_internal.h`: `last_error` keeps why registration failed (rejected key, waiting for approval, the server's reason), and is cleared on connect, so the device screen can show it.
- `include/tailtiny_internal.h`: task stacks trimmed from 8/14/12/8 KB (net_io, derp_tx, coord, wg_mgr) to 5/8/9/5 KB, the peaks measured on an AMOLED-1.75C plus 2-3 KB each. They live in internal RAM, which the audio path needs.
- `src/ml_wg_mgr.c`: the "WG peer UP" line in `ml_wg_mgr_peer_is_up` logs at debug, not info, because the port polls it every second.
