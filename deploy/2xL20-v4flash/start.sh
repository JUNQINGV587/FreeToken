#!/bin/bash
# DeepSeek-V4-Flash-0731-NVFP4 on 2x L20 (48GB).
# Container-internal command (docker inspect Config.Cmd); host-side docker run,
# per-flag rationale, measured geometry and rollback live in README.md.
echo "=== V4-Flash $(date -Is) ==="
free -g | head -2

export FREETOKEN_CT_PRE_ENSURE=1
export FREETOKEN_CT_REUSE_MIN=64
export FREETOKEN_CT_INLINE_MAX=2
export FREETOKEN_CT_A=0.05
export FREETOKEN_CT_B=0.30
export FREETOKEN_VRAM_ELASTIC=1

ft serve \
  --model /data/models/DeepSeek-V4-Flash-0731-NVFP4 \
  --host 0.0.0.0 --port 1925 \
  --tensor-parallel-size 2 --moe-ep-size 2 --gpu 0,1 \
  --moe-strategy offload \
  --moe-disk-tier on \
  --expert-ram-experts 176 \
  --moe-ram-pin-file /data/research/runs/20261008-v4flash-firstboot/pin_set_v4f_p176.json \
  --disk-tier-auto-reserve-gb 0 \
  --moe-cache-size 2500 \
  --memory-ratio 0.98 \
  --moe-cpu-tier \
  --num-tokens 98304 \
  --max-seq-len-override 1048576 \
  --max-output-tokens 131072 \
  --prefill-chunk-tokens 4096 \
  --prefill-chunk-adaptive \
  --max-running-requests 16 \
  --cuda-graph-max-bs 4
SERVE_PID=$!
trap 'kill -TERM "$SERVE_PID" 2>/dev/null' TERM INT
wait "$SERVE_PID"
