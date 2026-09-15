## Executive summary (read this first)

This plan is updated against the organiser repositories and public issue threads available on 2026-09-15. The winning submission should maximise expected production composite score while treating schema, cutoff, roster, and faithfulness failures as whole-unit losses. The current best route is a deterministic evidence pipeline with a strong target-aware prediction layer, empirical interval calibration, and a production-ready House API client; a LoRA adapter is an optional second lane, not a reason to delay the compliant baseline. Public smoke results are interface checks only because public units have no resolved outcomes and the production judge is not the smoke judge.

# Track 4 夺冠路径与执行计划（2026-09-15 版）

## 1. 官方状态核查与对策略的影响

本计划以官方 `upstream/main` 的最新公开内容为准，并以 [Track 4 官方仓库](https://github.com/Agenthon-2026/track4-analysis-public) 和 [共享 toolkit 仓库](https://github.com/Agenthon-2026/Agenthon2026-public) 为证据源。核查快照为 Track 4 `23da074`（2026-09-13）与共享仓库 `fbc57d29`（2026-09-12）；当前工作树 `6274b86` 落后于官方分支。截至本计划日期，官方最近变更包括：

- 评分器统一到 **3.1.0**，参与者应固定 `qfbench2-common` **v2.4.0**；不要继续使用旧的 v2.3.1。
- BYO 规则已经收敛为 **只提交一个 rank ≤ 64 的 LoRA adapter**，由主办方加载到其基础模型并提供 `MODEL_ENDPOINT`；禁止打包完整模型权重、完整微调模型或自行启动模型服务。`byo-large` / `byo-small` 是历史 descriptor 名称，不代表两档权重方案。
- House API 的已选配额是**每单位 25 次请求、每次最多 4000 输出 token**；输入限制以及失败/重试是否计费仍待平台公告，不能把 25 次当成可无条件重试预算。
- [离线训练政策](https://github.com/Agenthon-2026/track4-analysis-public/blob/23da0746010f19f21f8b180953477fcd19675351/docs/TRAINING-POLICY.md)允许合规的外部历史数据用于拟合、模型选择和区间校准，但每个任务使用的数据必须在其 cutoff 前可得，且必须提交 `ARTIFACT_PROVENANCE.md` 记录来源、许可证、首次可得时间和不可变版本。
- `build_smoke_verifier` 是不可排名的预览；生产评分只走 `build_verifier`。本地 lexical faithfulness、Development 结果和无 resolved outcome 的 smoke 分数都不能用于模型/提示词选优。
- 当前公开合同仍要求三类 target type（classification / regression / ranking）、完整 roster、每行有效 interval 和至少一个可解析 citation；任何单行 schema 问题都使整单位落到最坏分。
- 评分没有 interval width/sharpness 奖励；可操作目标是跨单位的 empirical coverage 接近卡片要求，而不是把区间做窄。

未冻结事项必须进入发布门禁：生产 NLI 的模型/Tokenizer revision、运行时与 cache digest；citation 阈值和 aggregate faithfulness 阈值的最终校准；House API 输入/失败/重试计费；正式 validation leaderboard 的开放时间与隐藏集摘要。相关公开讨论见 [issue #1](https://github.com/Agenthon-2026/track4-analysis-public/issues/1)、[issue #2](https://github.com/Agenthon-2026/track4-analysis-public/issues/2)、[issue #3](https://github.com/Agenthon-2026/track4-analysis-public/issues/3) 和 [issue #8](https://github.com/Agenthon-2026/track4-analysis-public/issues/8)。

## 2. 夺冠目标函数

把每个单位视为一次全有或全无的交付：

1. 先保证 g0 integrity、g1 schema、g2 cutoff/resource、g3 domain semantics 全部通过。
2. 再保证至少 80% roster entity 的 citation 能支持**提交的预测与区间**，而不是只支持 claim 文本。
3. 在合格单位上最大化 `0.70 × predictive_quality − 0.30 × |coverage − interval_level|`，并让失败单位数为零。

因此优先级固定为：**零整单位失败 > faithfulness 稳定过线 > 预测质量 > 区间校准的最后几个百分点 > 语言润色**。任何只提高平均分、却增加 schema 或 cutoff 风险的改动不接受。

### 作战判定（每个候选版本都必须回答）

- **可提交：** g0–g3 在所有回放单位均为 0 失败，输出可重放，且没有未解释的请求、证据或 cutoff 记录。
- **可排名：** 生产 judge 可用后，faithfulness 的下置信界仍高于 0.80；不能用 lexical smoke 值替代这个证据。
- **值得替换：** 新版本在固定 holdout 上的 predictive quality 改善，且其置信区间不与旧版本重叠到无法区分；同时失败率和 coverage 偏差不恶化。
- **立即回滚：** 任意整单位失败、出现一条 cutoff/引用越界，或请求账本无法解释实际配额消耗。

这些判定写入每次实验报告；没有报告的 prompt、检索、模型或 adapter 变更不进入候选提交。

## 3. 推荐技术路线

### A. 交付主线：House API + 确定性 evidence pipeline

保留当前 `strong_rag_baseline` 的可复现骨架，但把“示例 baseline”升级为提交候选：

- 启动时读取并校验 `task.json`、`card.toml`、manifest 和 corpus；按 cutoff 过滤文档，拒绝日期缺失或越界证据。
- 对每个实体建立实体别名、数值字段和表格单元的索引；BM25/词法检索作为必有路径，模型只做重排、数值解释和不确定性判断。不要依赖在线 embedding 下载。
- 让模型返回受限 JSON 中间结果：target type、预测值、理由、证据候选、区间依据；本地 validator 再生成最终 `answer.json`。模型输出不直接落盘。
- citation 由代码从已检索 span 生成，严格保存 `doc_id`、半开区间 `span_start/span_end`，并在提交前逐条重放 `tau_citation` 检查。
- 对 classification 使用任务 label vocabulary；对 regression 使用带单位的数值；对 ranking 以 `point_forecast` 排序，`rank` 只有在完整 permutation 时才输出。
- 使用离线 walk-forward / leave-one-cutoff-out 评估做模型选择和区间校准；不使用任何 cutoff 之后的答案、修订值或隐藏单位线索。

### B. 可选增强线：LoRA adapter

只有在 A 线通过完整合同测试、且已有可复现实验证据表明 adapter 提升 predictive quality 时才启用 B 线。adapter 线必须满足：一个 adapter 文件对、rank ≤ 64、固定 revision、不可启动模型服务、通过 `ARTIFACT_PROVENANCE.md` 证明训练/选择/校准 cutoff 合规。若生产 endpoint、adapter 加载或 bit-reproducibility 仍未开放验证，提交 A 线，避免把合规风险换成理论上限。

### C. 低成本保险：模型不可用降级

在 `MODEL_ENDPOINT` 不可达、超时或预算耗尽时，仍输出完整且可解析的 deterministic answer：实体全覆盖、合法 point forecast、保守区间和本地证据。降级路径只用于避免整单位失败；它不应被误当成夺冠模型，且必须在 `--network=none` 下反复验证。

## 4. 分阶段执行与退出门槛

### 阶段 0：合同锁定（1 天）

- 将工作树同步到官方最新提交，固定 toolkit v2.4.0 和 scorer 3.1.0。
- 生成 `submission.json`、`ARTIFACT_PROVENANCE.md`、镜像 digest 和依赖锁文件。
- 建立“官方变更监视表”：每周三 release 检查 scorer、judge、配额和 submission contract；变更后先跑合同回归再调模型。

**退出门槛：** `pytest scoring/ faithfulness/`、public-safe firewall、Docker smoke、`--network=none` 全部通过；无未解释的 schema 或 roster 差异。

### 阶段 1：证据与合规底座（2–3 天）

- 为全部公开单位建立离线索引和 evidence trace；逐行验证 span 可重放、日期不越 cutoff、实体绑定正确。
- 编写 adversarial fixtures：缺实体、重复实体、未知 label、rank 重复、NaN、空 claims、错误 interval level、越界 citation。
- 记录每次运行的请求数、token 数、seed/temperature、模型名和输入哈希，便于 House API 账本核对。

**退出门槛：** 机械错误率为 0；所有故意破坏样例都在本地被 g1/g2/g3 拒绝；任何失败都不能静默缩小 denominator。

### 阶段 2：预测与校准（3–5 天）

- 用 cutoff-aware 的历史切分比较：纯表格模型、词法 RAG、House API、以及（若合规）LoRA adapter。
- 对三类 target type 分开调参；不要把 classification 的 label accuracy、regression 的 skill score、ranking 的 Spearman 混为一个训练目标。
- 用 out-of-fold 结果校准 90% interval；先保证 coverage，再在 coverage 达标后改善 point forecast。区间宽度本身不加分。
- 将每个候选版本冻结成可回放报告，至少包含按 target type、prediction family、cutoff 年份和 evidence 命中情况的分层结果。

**退出门槛：** 在未见 cutoff 的回放集上，预测质量的置信区间下界超过 text-blind baseline 和 shipped minimal RAG；coverage 偏差容差必须在实验开始前登记，不得看结果后改；faithfulness 采用生产 judge 可用后再做最终调参。

### 阶段 3：生产化与竞赛提交（1–2 天）

- 只使用生产合同允许的 API；关闭 vendor-side web/search/code/retrieval tools。
- 预留请求预算：25 次配额按“每实体批处理 + 失败重试上限”分配，输入计费规则公布前不做激进重试。
- 做一次冷启动、一次网络受限、一次模型超时、一次重复运行；核对输出、日志、镜像 digest 和 provenance。
- validation leaderboard 开放后，先提交 A 线作为基线，再以单变量实验比较 prompt、retrieval、calibration 和 adapter 版本；禁止同时改多个组件导致不可归因。

**提交门槛：** 所有单位零 g0–g3 失败；生产 faithfulness ≥ 0.80；重复运行满足官方可复现规则；不存在 cutoff、工具调用、完整权重或 provenance 缺口。

## 5. 每周决策规则

- **官方 release 改 scoring/judge/schema：** 立即冻结模型调参，先更新 toolkit、合同测试和 golden fixtures，再恢复实验。
- **只有 smoke 分数变化：** 不改变夺冠方向；smoke 只用于接口诊断。
- **预测质量提升但 faithfulness 下降：** 先修证据绑定和假设生成，宁可暂缓提交。
- **House 配额不足：** 增加批处理、缓存和本地确定性解析；不得通过并发重试赌平台尚未公布的计费细则。
- **adapter 只在离线回放提升：** 保留为候选，不替换 House 主线，直到生产 adapter endpoint 和 bit-reproducibility 验证完成。

## 6. 现阶段立即行动清单

1. 对 `upstream/main` 做差异审计并建立独立同步提交；保留本地实验分支和未跟踪文件，不做无审计的强制覆盖。确认 `scripts/smoke-all-units.sh` 是否属于本次交付后再决定是否纳入镜像。
2. 将当前 baseline 的每次模型调用收敛为可计数的 request ledger，并补齐 25-request budget guard。
3. 把 interval 校准从固定窄带改成 cutoff-aware empirical calibration；无证据时使用覆盖优先的保守区间。
4. 完成三类 target type 的分层回放报告，报告中同时列出失败单位数和 faithfulness 诊断，不用一个总平均数掩盖整单位失败。
5. 准备 adapter 实验分支和 provenance，但在官方生产 judge、验证入口及 BYO operational contract 完整发布前，不将其作为唯一提交路径。
6. 订阅官方仓库 release/issue 更新；每周三 release 后 24 小时内完成一次合同差异审计。

## 7. 冻结前的决策树

1. **生产 judge 已冻结且可运行：** 立即建立真实 judge 的回归集，先校准 citation 与 faithfulness，再比较模型和 prompt。
2. **judge 已冻结但尚不可运行：** 只做 schema、cutoff、检索和预测质量工作；不把任何 faithfulness 数字写成排名预测。
3. **judge 或配额合同发生变化：** 立即停止当前候选的排名结论，重新生成合同 fixture 和请求预算；旧报告只能保留为历史记录。
4. **validation leaderboard 开放：** 先提交最小变更的 A 线取得基线，再按单变量顺序测试检索、提示词、校准和 adapter；每次提交记录镜像 digest、代码版本和请求账本。

计划负责人应在每次官方 release 后更新本节状态；若状态无法判断，默认按更严格的上一条处理。

这份计划的成功标准不是“公开 smoke 看起来更高”，而是：在官方合同冻结后，用一次可复现、零整单位失败的提交，把 evidence faithfulness、target-aware prediction 和 empirical coverage 同时带入可排名区间。
