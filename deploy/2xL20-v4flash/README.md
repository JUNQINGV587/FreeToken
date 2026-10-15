# DeepSeek-V4-Flash-0731-NVFP4 on 2x L20: 起服配方

一句话: 这套参数在 2x L20 48GB + 125 GiB RAM 上把**单流稳态做到 48.8 ms/token (20.4 tok/s)**,
**3 路并发 21.29 tok/s**, 并通过**真 3x24K tokens 长上下文**(3 路全部成功, 183.2s, 每路 24165 prompt tokens);
多让出 100 槽/卡把**单请求 prompt 顶到 KV 池上限**(实测 94612 tokens 通过, 2500 槽版只能到约 64K)。

定稿: 2026-10-15(2400 槽版)。同日更早的 2500 槽版单流更快(46.1 ms)但单请求上限低, 两者对照与回滚见 §10。
本目录 `start.sh` 是容器内命令, 宿主侧 `docker run` 见 §3。

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

**重建时先停旧容器**, 否则新容器 `docker run` 会在端口上失败
(`Bind for 0.0.0.0:1925 failed: port is already allocated`), 而旧容器还在正常服务,
很容易误判成"部署成功"。步骤: `docker stop <旧> && docker rename <旧> <旧>-old-<ts>` 之后
再跑上面的命令; 想留回滚就 rename 而不是 rm。

## 4. 参数与理由

| 参数 | 值 | 为什么 |
|---|---|---|
| `--tensor-parallel-size` / `--moe-ep-size` | 2 / 2 | 2 张卡; owner EP 要求 TP2+EP2 |
| `--moe-strategy` | `offload` | 专家集 145.125 GiB 远超 96 GiB 总显存, 必须 offload |
| `--moe-disk-tier` | `on` | 尾部按需从盘取; 与 hybrid 互斥(见 §9) |
| `--expert-ram-experts` | 176 | 宿主内存 pin 预算(host-wide, 按 rank 均分 = [88,88]/rank) |
| `--moe-ram-pin-file` | `pin_set_v4f_p176.json` | pin 集合; pin 失败是硬启动失败, 无自动回退 |
| `--moe-cache-size` | **2400** | VRAM 槽位数(1 槽 = 1 行专家 = 13.5 MiB); **单流与 3 路都随它近似线性变快**。比 2500 少 100 槽 = 每卡多让 1.33 GiB 余量, 换来单请求到池上限(§7/§8) |
| `--memory-ratio` | 0.98 | 显存预算系数 |
| `--num-tokens` | **98304** | **硬钉 KV 容量**。KV 池大小对单流与 3 路吞吐都无影响(§7), 所以把它压到刚好覆盖 3x24K 探针的量, 把显存让给槽位 |
| `--max-seq-len-override` | 1048576 | 单请求上下文上限, 与 KV 池容量无关 |
| `--max-output-tokens` | 131072 | |
| `--disk-tier-auto-reserve-gb` | 0 | 只在 auto 模式生效; 显式槽位下本就无效 |
| `--swa-num-pages-override` | 不设 | 由引擎自算: 98304 令牌下得 154 window pages |
| `--prefill-chunk-tokens` / `--prefill-chunk-adaptive` | 4096 / 开 | 分块 prefill |
| `--max-running-requests` | 16 | 压到 4 实测无收益(§6), 且会把窗口池下限压小, 不采用 |
| `--cuda-graph-max-bs` | 4 | 单流与 3 路都 <= 4, 全在图内; 图覆盖对吞吐的影响已由 A/B 排除 |
| `--moe-cpu-tier` | 开 | **不能关**: 关掉后单流从 62.4 ms 掉到 66.9/65.3 ms(§6) |
| env `FREETOKEN_VRAM_ELASTIC=1` | | prefill 前释放、decode 少步后回温 |
| env `FREETOKEN_CT_*` | | CPU 专家层认领策略(实测 CPU-MoE 52-62 GB/s > PCIe H2D 12.3 GB/s) |

`--moe-collect-stats` 开着不花钱(实测 62.4 vs 62.3 ms), 需要看 miss 率时可以直接加。

## 5. 实测几何(起服日志原文)

```
owner EP prefill overlap enabled: slots [0, 256) of 2400 are borrowed as the 2-layer prefill buffer ring
disk tier: 88/256 experts/layer pinned in RAM (learned pin set from .../pin_set_v4f_p176.json); the rest fetched from /data/models/... on slot-cache miss
Allocating 98304 tokens for DSV4 KV cache (154 window pages), total = 3.19 GiB     (rank 0 与 rank 1 各一条)
Free GPU memory after capturing CUDA graphs: 2.96 GiB
API server is ready to serve on 0.0.0.0:1925                                          (150s)
```

2400 槽/rank 约 31.6 GiB, 其中 256 槽借给 prefill ring; 可用 2144 槽 >> 非 pin 行 1720, 尾部全驻留。
两卡显存账: 每卡 31.6(槽) + 3.19(KV) + 9.7(非专家权重与激活) + 0.77(图) ≈ 45.3 GiB, 余 2.96。

## 6. 实测记录(2026-10-15)

单流用 `ft-bench-single.py`(流式, 512 token, 报尾 128 token 的稳态; 256-token 全程平均会把冷启动
摊进去, 两者差 2 倍以上, 不能混用)。3 路用 `ft-bench3.py`(chat 端点, 256 max tokens)。
每格两轮, 数字为两轮值。

| 配方(槽位 / KV) | 图后余量 | 单流稳态 ms/token | 3 路聚合 tok/s | 真 3x24K |
|---|---|---|---|---|
| 2000 / 262144(上一版) | 2.91 GiB | 62.2 / 62.9 | 17.58(部署后 16.74) | 通过 187.2s |
| 2000 / 196608 | 5.02 GiB | 62.1 / 62.7 | 未测 | 未测 |
| 2256 / 196608 | 1.65 GiB | 52.9 / 53.7 | 未测 | 未测 |
| 2256 / 196608 + `--disk-fetch-workers 16` | 1.65 GiB | 53.4 / 53.4 | 未测 | 未测 |
| 2400 / 131072 | 1.90 GiB | 48.8 / 49.5 | 20.24 | 未测 |
| 2400 / 131072 + `--max-running-requests 4` | 1.90 GiB | 48.6 / 49.3 | 19.64 | 未测 |
| 2500 / 65536 + `--max-running-requests 4` | 2.70 GiB | 46.5 / 47.2 | 20.57 | 未测(KV 池容不下 3x24K) |
| 2500 / 98304 | 1.63 GiB | 46.1 / 46.1 | 20.64 | 通过 202.2s |
| **2400 / 98304(本配方)** | **2.96 GiB** | **49.0 / 48.8** | **21.29** | **通过 183.2s** |

同一张表读出来的四条:
- 单流随槽位变快: 2000 -> 2256 是 -15%, 2256 -> 2400 是 -8%, 2400 -> 2500 是 -5%, 边际约 3.3 ms/100 槽。
- **KV 池不影响速度**: 2000 槽时 262144 与 196608 是 62.2 与 62.1 ms, 完全相同。
- **取数并行度不影响速度**: `--disk-fetch-workers` 8 -> 16 是 53.4 与 53.4 ms。
- **`--max-running-requests` 4 无收益**(48.8 -> 48.6 ms, 3 路 20.24 -> 19.64), 不采用。
- 关 `--moe-cpu-tier`: 62.4 -> 66.9/65.3 ms, 慢 4%, 必须留着。
- `--moe-collect-stats`: 62.4 vs 62.3 ms, 免费; 打开后读到稳态 miss 率 5.1%、每层缺 0.157 行、
  每层活跃 3.078 行, 即每 token 约 6.75 行(91 MiB)要过 PCIe, 这就是 46-62 ms 的主要成分。
- `--moe-cache-policy` 保持 LRU: 同一 trace 离线回放, LFU 的拷贝量是 LRU 的 4.4 倍。

长上下文下的实测(单流、递进请求, 后一档命中前一档前缀, 所以 TTFT 只含增量 prefill):

| 请求 prompt_tokens | TTFT | 尾 64 稳态 |
|---|---|---|
| 31492(冷) | 154.2 s | 13.05 tok/s = 76.7 ms/token |
| 47356 | 78.2 s | 27.50 tok/s = 36.4 ms/token |
| 63244 | 78.4 s | 34.43 tok/s = 29.0 ms/token |

读法: prefill 约 200 tok/s(两次 15.9K tokens 的增量各花 78 s, 三档一致);
**decode 不随上下文变慢**——29-77 ms 的差别来自槽位被前一轮捂热没有, 不是上下文长度
(短上下文热态是 46 ms)。表里 prompt_tokens 按服务端 usage 回读, 比目标值大
(造 prompt 的脚本按 2.15 字符/token 估, 中文实际约 1.64)。

基准工具(在宿主机 `/data/build`, 不在本仓库): `ft-bench-single.py`(单流分段),
`ft-bench3.py`(1/3/8 路 chat), `ft-longctx.py`(`FT_CTX=52000` 字符 = 24165 tokens, `FT_N=3`)。
跨工具/跨端点的数字不可直接比较。

## 7. 三条规律

- **槽位数是唯一有效旋钮**: 单流(2000/2256/2400/2500 -> 62.2/53.3/48.8/46.4 ms)与
  3 路(1200/1700/2000/2106/2302 -> 11.26/15.04/17.58/18.16/20.51 tok/s, 约 0.0084 tok/s 每槽)
  都跟着它走; 每 100 槽约值 3.3 ms(单流)。**槽位不是白拿的**: 余量小的版本单请求长不了(§8),
  本配方就是拿 100 槽/卡换回了 3 万 tokens 的单请求长度。
- **显存预算里槽位与 KV 抢同一块地**: 每 1024 tokens 约 0.033 GiB/rank(2500 槽下实测口径)。
  所以"再加槽位"只能从 KV 池里拿, 这就是为什么本配方把 KV 压到 98304。
- **8 路并发未定论**: 同一配方不同轮次拿过 23.73 与 37.52, 2302 槽 + 72K KV 又拿过 42.23。
  本配方 8 路实测 16.07-16.90, 可能不如旧配方; 本条只作记录, 不作为选型依据。
- **decode 不随上下文长度变慢**: 31K/47K/63K prompt 的尾段稳态是 76.7/36.4/29.0 ms(§6),
  差别来自槽位冷热而非上下文。随上下文涨的是 **prefill(约 200 tok/s)** 和对显存余量的要求(§8)。

## 8. 余量是硬门槛(三个失败样本)

- **0.14 GiB**: 起服正常, 第一个请求即
  `torch.OutOfMemoryError: Tried to allocate 704.00 MiB. GPU 1 has a total capacity of 47.45 GiB of which 102.88 MiB is free.`,
  随后 `Backend supervisor: backend worker freetoken-TP1-scheduler exited`。
- **1.07 GiB**: 短基准(1/3/8 路)全部正常, 真 3x24K prefill 时容器死亡。
- **1.63 GiB + 单个 72K prompt**: 起服与短请求都正常, 单请求 72K tokens 的 prompt 在 prefill 阶段
  `torch.OutOfMemoryError: Tried to allocate 264.00 MiB. GPU 0 ... 160.88 MiB is free`
  (位置 `attention/dsv4_indexer.py:72 indexer_select_prefill`), 随后
  `Backend worker is gone and cannot be restarted`。同样的 72K 拆成 3x24K 并发则通过。
- **1.63 GiB(2500 槽版)**: 真 3x24K 探针通过(202.2s); 单请求 63244 tokens 通过。
- **2.96 GiB(本配方)**: 真 3x24K 探针通过(183.2s); 单请求 **84292 与 94612 tokens 都通过**
  (94612 = 池子 98304 的 96%)。
- **2.91 GiB(更早的 2000 槽版)**: 探针运行中一度降到 2.09 GiB, 通过。

推论: 短请求只需 >0.2 GiB; 长 prefill 需要 1.6 GiB 以上, 单请求越长要得越多。
**单请求 prompt 的上限由余量决定**: indexer 的 top-k 临时张量按单序列长度线性涨(约 3.7 KiB/token),
`--prefill-chunk-tokens` 切不动它——1.63 GiB 余量下 63244 通过、约 72K OOM; 2.96 GiB 下 94612 通过,
顶到 KV 池上限。想再长只能继续让余量(减槽位)或加大池子, 两者都吃显存。根本解法在代码:
让 `indexer_select_prefill` 按 chunk 分片(未开工)。

## 9. 已知 CLI 与流程陷阱(期间踩过)

- `--num-tokens` 才是 KV 硬容量(dest 为 `num_token_override`), 与 `--num-pages` 互斥;
  `--num-token-override` **这个参数名不存在**, 会直接 `unrecognized arguments` 退出。
- `--kv-reserve-tokens` 只是**下限**: 显式 `--moe-cache-size` 时规划器把剩余预算全划给 KV
  (请求 262144 实际得到 347392, 余量 0.14 GiB, 首请求崩)。显式槽位务必配 `--num-tokens`。
- `--moe-cache-auto` 依据**启动时实测空闲显存**做预算, 宿主上另有进程占显存时结果不可复现
  (曾因此得到 2302 槽 / KV 72064 的偏差几何)。
- 显式槽位数**不做**预算校验: 给多了不报错, 只在运行期 OOM。
- **就绪判据用日志行**: `/v1/models` 在引擎加载完成前就会返回 200, 拿它当就绪探针会得到
  `HTTP 503` 的假失败; 等 `API server is ready to serve` 那一行。实例重启时还要比对这一行的
  **条数**再增长, 否则读到的是上一轮日志。
- **重建 = 先停旧容器**(见 §3), 否则端口冲突; 端口冲突时旧实例仍在服务, 别把它当成新实例。
- `--moe-strategy hybrid` **不能与本配方的 disk tier 同时用**: 引擎直接拒绝
  (`--moe-disk-tier requires the gpu decode path`), 而未 pin 的 1720 行只能从盘来, 所以 hybrid 走不通。
- `--disable-moe-prefill-overlap` 是反向杠杆: 关掉后 prefill 退化成整层拷贝, 单流 8.68 -> 6.10,
  3 路 17.58 -> 9.10。保持默认开启。

## 10. 回滚与备选几何

本配方只改参数, 不产生代码改动。回滚 = 把下面两项改回上一版, 然后 `docker rm -f freetoken-dsv4f`
再跑一遍 §3 的 `docker run`(重建前先按 §3 停旧容器):

| 参数 | 本配方(2400) | 2500 槽版 | 3 路版(2026-10-14) | 出厂版 |
|---|---|---|---|---|
| `--moe-cache-size` | 2400 | 2500 | 2000 | 1200 |
| `--num-tokens` | 98304 | 98304 | 262144 | 1048576 |
| `--swa-num-pages-override` | 不设 | 不设 | 不设 | 435 |
| `--cuda-graph-max-bs` | 4 | 4 | 4 | 16 |
| `--memory-ratio` | 0.98 | 0.98 | 0.98 | 0.95 |
| `--disk-tier-auto-reserve-gb` | 0 | 0 | 0 | 8 |

单流最快的是 2500 槽版(46.1 ms)但单请求只能到约 64K; 要 3 路优先可回 3 路版(17.58 tok/s, 上下文 262144)。

备选几何: **2400 槽 / 131072 tokens**(单流 48.8 ms、3 路 20.24、余量 1.90 GiB、上下文 131K tokens)。
与本配方槽位相同、池子大 33%, 但余量少 1.06 GiB ⇒ 单请求上限反而低。**2400/131072 没跑过真 3x24K 探针**;
本配方(2400/98304)是实测过的组合, 优先用。
更省事的回滚做法是重建前先存档当前实例(`docker stop` + `docker rename`), 之后一条 `docker rename` 换回来;
本仓库不假定存档容器一定存在(它可能被清理), 所以上表才是权威口径。宿主上的脚本备份
`/data/build/start_ftv4f.sh.bak-<ts>` 只作参考, 不在仓库里、不保证存在。

## 11. 验收清单

```bash
docker logs freetoken-dsv4f 2>&1 | grep -E "slots \[0, 256\) of|Allocating .* KV cache|Free GPU memory after"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:1925/v1/models
docker cp /data/build/ft-bench-single.py freetoken-dsv4f:/tmp/ && docker exec -e FT_PORT=1925 -e FT_GEN=512 -e FT_TAIL=128 freetoken-dsv4f python3 /tmp/ft-bench-single.py
docker cp /data/build/ft-longctx.py    freetoken-dsv4f:/tmp/ && docker exec -e FT_PORT=1925 -e FT_N=3 -e FT_CTX=52000 freetoken-dsv4f python3 /tmp/ft-longctx.py
```

四条几何 + 200 + 单流尾段(应约 20.4 tok/s = 48.8 ms) + 3x24K 探针, 全部满足才算这次部署成立。

## 12. 遗留待办

- **TTFT 仍是 5-7 s**(单流首 token; 长上下文下更贵: 84292 tokens 冷起首 token **379.6 s**,
  ≈222 tok/s, 命中前缀时只付增量、55.6 s)。已定位到机制: 该 prompt 在 prefill 阶段要把自己路由到的
  专家行整批搬进显存(约 2200 行/rank, 其中约 1/3 不在 pin 集里、要读盘), 与 KV 池、槽位缓存
  都无关(槽位从 2000 加到 2500, TTFT 从 8.12 到 8.15 s, 不动)。候选方向: 按 prefill 轨迹重学
  pin 集(`tools/trace/make_ram_pin_set.py`), 或做 prefill 感知的取数准入。未开工。
- **视觉版(DeepSeek-V4-Flash-Vision-Exp)没接上**, 三条路线都没走: (A) 对官方视觉版做无损
  mxfp4 -> NVFP4 容器改写(保留 `vision.*`/`aligner.*` 布局, 不动 fork), (B) 下载社区 NVFP4 视觉版
  并把 `encoder.*` 适配成 `vision.*`/`aligner.*`, (C) 给 disk tier 实现 MXFP4 读路径(只能 TP1, 更慢)。
  当前 fork 的形态(owner EP + disk tier)只接受 NVFP4 专家库, 而官方视觉版的专家是 MXFP4
  (`expert_dtype: fp4`, e8m0 每 32 元素), 所以直接换权重跑不起来。未开工。
