# DeepSeek-V4-Flash-0731-NVFP4 on 2x L20: 起服配方

一句话: 这套参数在 2x L20 48GB + 125 GiB RAM 上把 **3 路并发做到 17.58 tok/s**(旧配方 11.26 的 1.56 倍),
并让**真 3x24K tokens 长上下文稳定通过**(3 路全部成功, 187.2s, 每路 24165 prompt tokens)。

定稿: 2026-10-14。本目录 `start.sh` 是容器内命令, 宿主侧 `docker run` 见 §3。

## 1. 硬件与权重

| 项 | 值 |
|---|---|
| GPU | 2x NVIDIA L20, 47.45 GiB 可用/卡 |
| 宿主内存 | 125 GiB |
| 权重 | `/data/models/DeepSeek-V4-Flash-0731-NVFP4` |
| 专家规模 | 11008 行 x 13.5 MiB = 145.125 GiB (99072 tensor, 不含 MTP) |
| pin 文件 | `/data/research/runs/20261008-v4flash-firstboot/pin_set_v4f_p176.json` |

注意: 本实例与 V4.1 生产容器 `freetoken-dsv41f` **显存互斥, 不能同时起**(41 占约 34 GiB/卡, v4f 需约 45 GiB/卡)。

## 2. 镜像与代码

| 项 | 值 |
|---|---|
| 镜像 | `freetoken:v4f-syncport-c84ffb6ba8` |
| 源码 | commit `c84ffb6ba8`, 分支 `sync/v4f-upstream-20261014` |
| 说明 | 镜像 tag 即 commit 前 12 位; 构建时工作区干净, 镜像与该 commit 逐字对应 |

## 3. 起服(宿主侧)

```bash
docker run -d --name freetoken-dsv4f --restart unless-stopped \
  --gpus all -p 1925:1925 --shm-size 64m \
  -e DEBIAN_FRONTEND=noninteractive -e PYTHONUNBUFFERED=1 -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -e PIP_RETRIES=10 -e PIP_TIMEOUT=120 \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e CUDA_HOME=/opt/cuda -e LD_LIBRARY_PATH=/opt/cuda/lib64:/opt/cuda/lib \
  -v /mnt/virtualization/workspace:/data \
  -v /mnt/virtualization/appdata/freetoken-dsv4f/cache:/root/.cache \
  -v /mnt/virtualization/appdata/freetoken-dsv4f/triton:/root/.triton \
  freetoken:v4f-syncport-c84ffb6ba8 bash /data/build/start_ftv4f.sh
```

三个 bind 是基础设施(工作区 / 权重与 pip 缓存 / triton 内核缓存), 不参与显存切片;
换掉它们只会让内核重编、缓存重拉, 不会变快。

## 4. 参数与理由

| 参数 | 值 | 为什么 |
|---|---|---|
| `--tensor-parallel-size` / `--moe-ep-size` | 2 / 2 | 2 张卡; owner EP 要求 TP2+EP2 |
| `--moe-strategy` | `offload` | 专家集 145.125 GiB 远超 96 GiB 总显存, 必须 offload |
| `--moe-disk-tier` | `on` | 尾部按需从盘取 |
| `--expert-ram-experts` | 176 | 宿主内存 pin 预算(host-wide, 按 rank 均分 = [88,88]/rank) |
| `--moe-ram-pin-file` | `pin_set_v4f_p176.json` | pin 集合; pin 失败是硬启动失败, 无自动回退 |
| `--moe-cache-size` | 2000 | VRAM 槽位数(1 槽 = 1 行专家 = 13.5 MiB); 3 路吞吐与它近似线性, 见 §7 |
| `--memory-ratio` | 0.98 | 显存预算系数 |
| `--num-tokens` | 262144 | **硬钉 KV 容量**; 从 1M 缩到 256K 腾出 5.5 GiB 余量, 3 路速度几乎不损 |
| `--max-seq-len-override` | 1048576 | 单请求上下文上限, 与 KV 池容量无关 |
| `--max-output-tokens` | 131072 | |
| `--disk-tier-auto-reserve-gb` | 0 | 只在 auto 模式生效; 显式槽位下本就无效 |
| `--swa-num-pages-override` | 不设 | 由引擎自算: 256K 令牌下得 410 window pages |
| `--prefill-chunk-tokens` / `--prefill-chunk-adaptive` | 4096 / 开 | 分块 prefill |
| `--max-running-requests` | 16 | |
| `--cuda-graph-max-bs` | 4 | 3 路 <= 4 都在图内; 图覆盖对吞吐的影响已由 A/B 排除(见 §7 注) |
| env `FREETOKEN_VRAM_ELASTIC=1` | | prefill 前释放、decode 少步后回温 |
| env `FREETOKEN_CT_*` | | CPU 专家层(实测 CPU-MoE 52-62 GB/s > PCIe H2D 12.3 GB/s) |

## 5. 实测几何(起服日志原文)

```
owner EP prefill overlap enabled: slots [0, 256) of 2000 are borrowed as the 2-layer prefill buffer ring
Allocating 262144 tokens for DSV4 KV cache (410 window pages), total = 8.50 GiB
Free GPU memory after capturing CUDA graphs: 2.91 GiB
API server is ready to serve on 0.0.0.0:1925        (140s)
```

2000 槽/rank 约 26.4 GiB, 其中 256 槽借给 prefill ring; 可用 1744 槽 >= 非 pin 行 1720, 尾部全驻留。

## 6. 实测记录(2026-10-14)

| 配方 | 槽位 | KV | 图后余量 | 单流 | 3 路 | 真 3x24K |
|---|---|---|---|---|---|---|
| 部署前(旧) | 1200 | 1048576 / 13.97 GiB | 7.67 GiB | 7.15 | 11.26 | 不达标 |
| **本配方** | **2000** | **262144 / 8.50 GiB** | **2.91 GiB** | **8.68** | **17.58**(部署后复测 16.74) | **通过**(187.2s) |

基准工具(在宿主机 `/data/build`, 不在本仓库): `ft-bench3.py`(`/v1/chat/completions`, 256 max tokens, 15-token prompt),
`ft-longctx.py`(`FT_CTX=52000` 字符 = 24165 tokens, `FT_N=3`, 64 输出)。跨工具/跨端点的数字不可直接比较。

## 7. 两条法则与一条不等式

- **3 路并发吞吐由槽位数决定, 基本与 KV 池无关**:
  1200 -> 11.26, 1700 -> 15.04(KV 256K)/15.48(KV 1M), 2000 -> 17.58, 2106 -> 18.16, 2302 -> 20.51 tok/s,
  约 0.0084 tok/s 每槽。同为 1700 槽时, KV 从 256K 换到 1M(四倍), 3 路只差 3%。
- **8 路并发未定论**: 同一配方不同轮次拿过 23.73 与 37.52, 2302 槽 + 72K KV 又拿过 42.23。
  本配方 8 路实测 16.07, 可能不如旧配方(23.73); 本条只作记录, 不作为选型依据。
- **不等式**: 槽位与 KV 共用同一块显存。2000 槽 + KV 1M 需要 `2.91 - 5.47 = -2.56 GiB`, 不成立。
  按「3 路够用」选本配方; 若改为 8 路优先, 必须牺牲槽位(1600 槽 + KV 1M: 3 路约 13.9, 8 路约 33)。

## 8. 余量是硬门槛(两个失败样本)

- **0.14 GiB**: 起服正常, 第一个请求即
  `torch.OutOfMemoryError: Tried to allocate 704.00 MiB. GPU 1 has a total capacity of 47.45 GiB of which 102.88 MiB is free.`,
  随后 `Backend supervisor: backend worker freetoken-TP1-scheduler exited`。
- **1.07 GiB**: 短基准(1/3/8 路)全部正常, 真 3x24K prefill 时容器死亡。
- **2.91 GiB(本配方)**: 探针运行中一度降到 2.09 GiB, 通过。

推论: 短请求只需 >0.2 GiB; 长 prefill 需要 2 GiB 以上。余量不足不会在短对话里暴露。

## 9. 已知 CLI 陷阱(期间踩过)

- `--num-tokens` 才是 KV 硬容量(dest 为 `num_token_override`), 与 `--num-pages` 互斥;
  `--num-token-override` **这个参数名不存在**, 会直接 `unrecognized arguments` 退出。
- `--kv-reserve-tokens` 只是**下限**: 显式 `--moe-cache-size` 时规划器把剩余预算全划给 KV
  (请求 262144 实际得到 347392, 余量 0.14 GiB, 首请求崩)。显式槽位务必配 `--num-tokens`。
- `--moe-cache-auto` 依据**启动时实测空闲显存**做预算, 宿主上另有进程占显存时结果不可复现
  (曾因此得到 2302 槽 / KV 72064 的偏差几何)。
- 显式槽位数**不做**预算校验: 给多了不报错, 只在运行期 OOM。

## 10. 回滚

```bash
docker stop freetoken-dsv4f
docker rename freetoken-dsv4f-old-20261014-2318 freetoken-dsv4f
docker start freetoken-dsv4f
```

存档容器名带日期; 旧启动脚本备份为 `start_ftv4f.sh.bak-<ts>`。

## 11. 验收清单

```bash
docker logs freetoken-dsv4f 2>&1 | grep -E "slots \[0, 256\) of|Allocating .* KV cache|Free GPU memory after"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:1925/v1/models
docker cp /data/build/ft-bench3.py  freetoken-dsv4f:/tmp/ && docker exec -e FT_PORT=1925 -e FT_LEVELS=1,3 -e FT_GEN=256 freetoken-dsv4f python3 /tmp/ft-bench3.py
docker cp /data/build/ft-longctx.py freetoken-dsv4f:/tmp/ && docker exec -e FT_PORT=1925 -e FT_N=3 -e FT_CTX=52000 freetoken-dsv4f python3 /tmp/ft-longctx.py
```

三条几何 + 200 + 3 路短基准 + 24K 探针, 全部满足才算这次部署成立。
