# 芯封智检 · 多智能体协同质检系统

基于 AIP（Agent Interconnection Protocol）国家标准实现的多智能体芯片封装缺陷检测系统。
四个能力型智能体 + 一个主控编排智能体，通过 AIP JSON-RPC over mTLS 互联，可在梧桐智能体互联平台上完成注册、发现、互联与访问。

## 一、架构

```
                    ┌──────────────────────────┐
   image url / base64│   主控智能体 (Leader)      │  AIP /rpc + mTLS
  ─────────────────► │  chip-quality.orchestration│
                    └───────────┬──────────────┘
                                │ 依次编排
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │ 图像预处理    │      │ 缺陷检测      │      │ 质检报告      │
  │ preprocess   │      │ detect       │      │ report       │
  └──────────────┘      └──────────────┘      └──────────────┘
          │                     │                     │
          └─────────────────────┴─────────────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ 数据回流智能体 (feedback)  │
                    └──────────────────────────┘
```

每个智能体具备：独立 AIC、独立 mTLS 证书、独立服务端点、独立 AMP 日志。

## 二、能力

| 智能体 | 技能 ID | 职责 |
|---|---|---|
| 主控 | chip-quality.orchestration | 接收图像，编排预处理→检测→报告，汇总结果 |
| 预处理 | chip-image.preprocess | 多格式解码、自动裁剪、缩放与留白处理 |
| 检测 | chip-inspection.defect-detection | YOLO 缺陷检测，输出缺陷类别、置信度与坐标 |
| 报告 | chip-quality.report | 依据检测结果生成缺陷清单、统计与判定 |
| 回流 | chip-quality.feedback | 保存人工复核样本、维护统计、达阈值触发重训请求 |

协议能力：AIP 02.02，方法覆盖 start / get / continue / complete / cancel。

## 三、运行要求

| 项 | 要求 |
|---|---|
| Python | 3.11 |
| 依赖 | acps-sdk 2.2.0、fastapi、uvicorn、httpx、pillow、pyyaml |
| 检测框架 | ultralytics（可选，仅在启用检测时需要） |
| 传输安全 | 服务端证书、客户端证书、信任库（PEM） |
| 证书 CN | **必须是 10 段式 AIC**，SDK 校验段数，不足 10 段返回 `-32008 AuthenticationRequiredError` |

模型权重不随仓库提供，需自行准备并在配置中指定路径。

## 四、快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**acps-sdk 未发布到 PyPI**，需从 AIP 开源社区单独安装：

```bash
git clone https://atomgit.com/AIP-PUB/ACPs-community.git
.venv/bin/python -m pip install ./ACPs-community/acps-sdk

# 或直接从仓库子目录安装
.venv/bin/python -m pip install "git+https://github.com/AIP-PUB/ACPs-community.git#subdirectory=acps-sdk"
```

该 SDK 使用木兰宽松许可证（MulanPSL-2.0），本仓库不包含其源码。

### 2. 准备证书

将平台签发的以下文件放入 `certs/` 目录：

```
certs/
  leader-cert.pem        # 主控客户端证书（含中间证书）
  leader-key.pem
  leader-server-cert.pem # 主控服务端证书
  leader-server-key.pem
  {agent}-server-cert.pem
  {agent}-server-key.pem
  trust-bundle.pem       # 根证书 + 中间证书
```

### 3. 配置环境变量

| 变量 | 说明 | 示例 |
|---|---|---|
| CHIP_MTLS_ENABLED | 是否启用 mTLS | 1 |
| CHIP_LEADER_AIC | 主控 AIC | 1.2.156.3088.1.BUPT.XXXXXX.XXXXXX.XXXXXX.XXXX |
| CHIP_LEADER_CERT | 主控客户端证书路径 | certs/leader-cert.pem |
| CHIP_LEADER_KEY | 主控私钥路径 | certs/leader-key.pem |
| CHIP_TRUST_BUNDLE | 信任库路径 | certs/trust-bundle.pem |
| CHIP_AIC_PREPROCESS | 预处理 AIC | 同上格式 |
| CHIP_AIC_DETECT | 检测 AIC | 同上格式 |
| CHIP_AIC_REPORT | 报告 AIC | 同上格式 |
| CHIP_AIC_FEEDBACK | 回流 AIC | 同上格式 |
| CHIP_MODEL_PATH | 检测权重路径 | models/best.pt |
| CHIP_URL_TOKEN | 网页访问令牌（可选） | 随机字符串 |
| CHIP_PUBLIC_BASE | 对外基础地址（用于图片分享） | https://your-host |

### 4. 启动

```powershell
# Windows：按 mTLS 模式启动五个服务
./scripts/start-mtls.ps1

# 或分别启动
python agents/secure_server.py preprocess
python agents/secure_server.py detect
python agents/secure_server.py report
python agents/secure_server.py feedback
python agents/secure_server.py leader
```

### 5. 验证

```bash
# 健康检查（需客户端证书）
curl --cert certs/leader-cert.pem --key certs/leader-key.pem \
     --cacert certs/trust-bundle.pem https://127.0.0.1:8017/health

# 未携带证书应被拒绝
curl -k https://127.0.0.1:8017/health
```

## 五、目录结构

```
chip-inspect-agents/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  agents/            四个能力型智能体、mTLS 启动器、AMP 消息模块
  chipinspect/       检测与主控包（server 编排、模型推理、预处理、基准测试）
  acs/               五个 ACS 元数据示例
  scripts/           启动、守护、日志转发、ACS 校验脚本
  tests/             端到端测试、心跳守护、重训脚本
  docs/              架构、AIP 接入、部署说明
```

## 六、不入库内容

以下内容因隐私或安全原因不随仓库提供：

- 模型权重（*.pt / *.pth / *.onnx）
- 真实产线样本与标注数据
- 证书、私钥、信任库
- 访问令牌、frp 配置、账号信息

仓库中的 ACS 为示例结构，AIC 与端点需替换为你自己的值。

## 七、设计说明

1. **每个智能体独立身份**：独立 AIC 与证书，请求中的 senderId 必须与证书 CN 一致，防止身份冒用。
2. **主控编排、子智能体无状态**：预处理与检测不保存会话，便于横向扩展。
3. **双形态输出**：检测智能体同时返回结构化数据与纯文本 JSON，兼容不支持结构化解析的调度器。
4. **宽容入参**：每个智能体同时接受 base64、本地路径、公网图片链接与说明文字中内嵌的链接。
5. **可观测**：access / message / heartbeat 三类日志，messageId 与 taskId 对齐，便于平台按任务配对统计。

## 八、已知边界

- 业务链路为主控驱动的星型编排，子智能体之间不主动互相调用。
- 平台发现服务按技能名称建索引，按智能体显示名称检索命中率为 0。
- 部署示例基于单机与内网穿透，生产环境建议独立网关与多实例。

## 九、许可

本项目采用 MIT 许可证，详见 LICENSE。
所依赖的 acps-sdk 遵循其上游许可，本仓库不包含其源码。
