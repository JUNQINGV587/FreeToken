# FreeToken browser playground

A dependency-free browser frontend (beyond FreeToken's existing server dependencies) for an
already-running FreeToken API. It proxies a fixed loopback upstream so the browser UI and API
share one origin without widening the engine's CORS policy.

It shows streaming chat, per-turn input/output tokens, TTFT, estimated prefill/decode throughput,
server-side live throughput, KV and MoE cache occupancy, RAM, and NVIDIA GPU telemetry.

```bash
python examples/web_playground/server.py \
  --host 127.0.0.1 \
  --port 30002 \
  --upstream http://127.0.0.1:1919
```

Open <http://127.0.0.1:30002/>. To use it from a trusted LAN, bind `--host 0.0.0.0` and open the
machine's LAN address instead.

The playground has no authentication. A LAN bind lets any reachable host submit inference
requests, so do not expose it directly to an untrusted network or the public internet.

The displayed per-turn prefill rate is `prompt_tokens / TTFT`, so queuing and HTTP latency are
included. Decode rate uses the interval between the first and last streamed output token.
