# AIP 接入说明

## 一、三层标识

| 标识 | 作用 | 来源 |
|---|---|---|
| ACS | 智能体元数据，声明能力、端点、安全方案 | 由开发者编写，提交平台审核 |
| AIC | 智能体唯一标识，10 段式 | 平台审核通过后签发 |
| 证书 | mTLS 身份凭证，CN 必须等于 AIC | 平台 CA 签发 |

三者必须一致。证书 CN 与 AIC 不符会导致握手被拒或身份绑定失败。

**证书 CN 必须是 10 段式 AIC，缺一段都不行。** SDK 在服务端会校验段数，不满足时直接拒绝请求：

```json
{"jsonrpc":"2.0","id":"1","result":null,
 "error":{"code":-32008,"message":"AuthenticationRequiredError",
          "data":"invalid peer certificate CN: AIC 必须包含 10 段，当前为 1 段"}}
```

正确示例（10 段）：

```
1.2.156.3088.1.BUPT.29ZHHL.GJZM4A.2HHVSC.0AH4
```

本地自签证书做测试时，CN 也要凑满 10 段，否则连健康检查之外的任何 RPC 都过不去：

```
1.2.156.3088.1.TEST.TEST01.TEST02.TEST03.0TST
```

## 二、ACS 关键字段

| 字段 | 说明 | 常见错误 |
|---|---|---|
| aic | 智能体标识 | 留空或写占位符 |
| protocolVersion | 协议版本 | 写成 2.2 而非 02.02 |
| endPoints | 服务端点数组 | 键名大小写错误 |
| securitySchemes | 安全方案 | 声明 mutualTLS 但实际跑 HTTP |
| capabilities | 能力声明 | 与实际行为不符 |
| skills | 技能列表 | 缺少 id，导致发现服务无法索引 |

## 三、服务端必须实现的方法

| 方法 | 语义 |
|---|---|
| start | 创建任务，返回快照 |
| get | 查询任务状态 |
| continue | 在等待输入后继续任务 |
| complete | 结束任务 |
| cancel | 取消任务 |

只实现 start 会导致追问机制失效：平台在收到 awaiting-input 后，会用 continue 把用户回复发回来。
若 continue 走默认实现（只记录历史、不执行业务），任务会永远停在等待输入。

## 四、状态机

```
accepted → awaiting-input → awaiting-completion → completed
                │                   │
                └── continue ───────┘
                        │
                    failed / cancelled
```

要点：

- 缺参数时返回 awaiting-input，并给出可操作的提示
- 收到 continue 后必须真正执行业务，不能只回显原任务
- 任务结束用 awaiting-completion 表示结果就绪，调用方需再发 complete

## 五、mTLS 配置

```python
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(server_cert, server_key)   # 服务端证书需含中间证书
ctx.load_verify_locations(trust_bundle)        # 根证书 + 中间证书
ctx.verify_mode = ssl.CERT_REQUIRED            # 强制校验客户端证书
ctx.minimum_version = ssl.TLSVersion.TLSv1_2
```

常见故障：

| 现象 | 原因 |
|---|---|
| 握手失败 | 信任库缺中间证书 |
| 平台侧 mTLS connection failed | 服务端拒绝客户端证书，或反向被 URL token 拦截返回 403 |
| 身份绑定失败 | 请求 senderId 与证书 CN 不一致 |

## 六、身份绑定

请求体的 senderId 必须与客户端证书 CN 一致。实现方式：从 TLS 连接取出对端证书，解析 CN，与请求中的 senderId 比对。

注意：这里的 CN **必须是 10 段式 AIC**，SDK 会校验段数。不足 10 段时返回 `-32008 AuthenticationRequiredError`，提示「AIC 必须包含 10 段」。自签证书测试时也要凑满 10 段。

## 七、发现服务

```http
POST https://wt.ioa.pub/discovery/acps-adp-v2/discover
{"type": "explicit", "query": "技能名称", "limit": 20}
```

注意事项：

- 平台按技能名称建索引，按智能体显示名称检索命中率为 0
- 按 AIC 精确检索不稳定
- 返回结构中 acsMap 为 AIC 到 ACS 的映射，需自行按 skills[].id 二次过滤

## 八、AMP 上报

| 主题 | 内容 |
|---|---|
| amp.access | 调用记录 |
| amp.message | 消息生命周期 |
| amp.heartbeat | 心跳 |

按任务配对统计的规则：access 与 message 的 correlationId 都填 taskId；message 的 send 与 receive 使用相同 messageId。
