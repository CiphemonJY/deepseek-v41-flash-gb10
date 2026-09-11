#!/usr/bin/env python3
"""garble_gate.py capture|compare [BASE_URL]  -- greedy reference gate (token-level), the reference's postserve idea.
capture: run the fixed prompt set at temperature 0 with logprobs, save tokens per prompt to ~/dsv41/greedy_baseline.json
compare: rerun, report per-prompt exact-match / prefix-match against the baseline, repeated-token runs, and token ids.
Exit 0 only if no prompt shows the garbage signature (U+FFFD / repeated-id run >= 8 / non-printable / empty) and prompts 0,1,3 match the baseline exactly."""
import json, sys, os, urllib.request, collections
mode = sys.argv[1]; BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8888/v1"
M = "deepseek-v4-flash-dspark"; OUT = os.path.expanduser("~/dsv41/greedy_baseline.json")
PROMPTS = [
 ("chat", "Count from 1 to 30, comma separated."), ("chat", "List the first 25 prime numbers, comma separated, then say DONE."),
 ("chat", "In one sentence, what is tensor parallelism?"), ("chat", "中国的首都是哪里？只回答城市名。"),
 ("chat", "Write a haiku about a mountain lake."), ("chat", "Explain in three sentences why the sky is blue."),
 ("raw", "The capital of France is"), ("raw", "Once upon a time, in a small village by the sea,"),
]
def run(kind, prompt, n=48):
    if kind == "chat":
        body = {"model": M, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "logprobs": True, "top_logprobs": 1}
        url = BASE + "/chat/completions"
    else:
        body = {"model": M, "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": 1}
        url = BASE + "/completions"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    ch = d["choices"][0]
    if kind == "chat":
        toks = [c["token"] for c in (ch.get("logprobs") or {}).get("content") or []]; text = ch["message"].get("content") or ""
    else:
        toks = (ch.get("logprobs") or {}).get("tokens") or []; text = ch.get("text") or ""
    return toks, text
res = {}
for i, (k, p) in enumerate(PROMPTS):
    toks, text = run(k, p); res[str(i)] = {"kind": k, "prompt": p, "tokens": toks, "text": text}
if mode == "capture":
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1); print(f"baseline captured: {len(res)} prompts -> {OUT}")
    for i, r in res.items(): print(f"  [{i}] {r['kind']:4s} {len(r['tokens'])} toks :: {r['text'][:70]!r}")
    sys.exit(0)
base = json.load(open(OUT)); ok_all = True
for i, r in res.items():
    b = base[i]["tokens"]; t = r["tokens"]
    pre = 0
    for x, y in zip(b, t):
        if x != y: break
        pre += 1
    runs = max((len(list(g)) for _, g in __import__("itertools").groupby(t)), default=0)
    exact = (b == t); text = r["text"]
    printable = (sum(ch.isprintable() or ch in (chr(10) + chr(9)) for ch in text) / len(text)) if text else 1.0
    # cross-mode tolerant: garbage = U+FFFD, repeated-id runs, non-printable text, or empty where the baseline was not.
    # Greedy divergence with coherent text is NOT garbage (graph kernels differ numerically). Exact-answer prompts 0,1,3 must match exactly.
    garbage = (chr(0xFFFD) in text) or runs >= 8 or printable < 0.9 or (not text.strip() and len(b) > 1)
    ok = (not garbage) and (exact if i in ("0", "1", "3") else True)
    ok_all &= ok
    print(f"  [{i}] {'OK ' if ok else 'BAD'} exact={exact} prefix_match={pre}/{len(b)} max_run={runs} :: {r['text'][:60]!r}")
print("GARBLE_GATE", "PASS" if ok_all else "FAIL"); sys.exit(0 if ok_all else 1)
