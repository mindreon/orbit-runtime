# AgentScope 2.0.9 实测脚本（2026-09-28）

06 §3 里 S2、S4、S5、S7、S8 和 SOP 逐步执行的结论，来自这些脚本的实际运行结果。它们是一次性的调研脚本，不是正式测试；Step 5 要把它们改写成 orbit-runtime 里的 characterization test。

运行方式：脚本里写死了 `/tmp/as_verify` 路径，先把本目录复制过去，再用 orbit-runtime 的 venv 运行。

```bash
mkdir -p /tmp/as_verify && cp *.py /tmp/as_verify/ && cd /tmp/as_verify
PY=<orbit-runtime>/.venv/bin/python
$PY s7a.py crash 1 && $PY s7a.py resume          # S7①：推理后、执行前的检查点
$PY s7a2.py first && $PY s7a2.py second && $PY s7a2.py third   # S7②：一批调用中间的检查点
$PY s7b.py crash && $PY s7b.py resume            # S7③：一轮结束后的检查点
$PY fmt.py                                       # S7②的上下文格式化后缺少 tool 结果
$PY s2.py park && $PY s2.py decide               # S2
$PY s2.py park && $PY s2p.py one && $PY s2p.py two   # S2：只批准一部分
$PY s4b.py                                       # S4：取消落在 on_acting middleware 里
$PY s4c.py                                       # S4：取消落在工具内部
$PY s2.py park && $PY intr.py                    # S8：停在审批时 UserInterruptEvent
$PY sop_step.py                                  # SOP 逐步执行
# S5：需要一个装了 agentscope==2.0.8 的 venv（uv venv v208 && uv pip install agentscope==2.0.8）
v208/bin/python s5.py dump && $PY s5.py load
```

`common.py` 提供一个按脚本返回工具调用的假模型（`ScriptModel`）和会计数的工具 `tool_a`、`tool_b`。

本目录是 orbit-infra `docs/architecture/evidence/as-2.0.9` 的拷贝：`tests/test_agentscope_characterization.py` 在独立的 orbit-runtime checkout（CI）里也要运行这些脚本，而 CI 取不到 orbit-infra。改脚本时两处一起改。
