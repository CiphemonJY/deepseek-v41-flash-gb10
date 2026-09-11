#!/usr/bin/env python3
"""battery.py (run on the head node): the council's 4-step battery against :8888. (a) greedy gate; (b) ~2000-token prompt; (c) 6 streams; (d) gate again."""
import json, os, subprocess, sys, time, urllib.request, concurrent.futures as cf
B = "http://127.0.0.1:8888/v1"; M = "deepseek-v4-flash-dspark"
def chat(content, n):
    body = json.dumps({"model": M, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0}).encode()
    t0 = time.time(); d = json.load(urllib.request.urlopen(urllib.request.Request(B + "/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    c = d["choices"][0]["message"].get("content") or ""; return c, d["usage"]["completion_tokens"], time.time() - t0
def gate(tag):
    r = subprocess.run([sys.executable, os.path.expanduser("~/dsv41/garble_gate.py"), "compare"], capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.startswith("GARBLE_GATE")]; bad = [l for l in r.stdout.splitlines() if " BAD " in l]
    print(f"[{tag}] {last[-1] if last else 'no verdict'} ({len(bad)} bad prompts)"); return r.returncode == 0
ok_a = gate("a: greedy gate")
para = "The quick brown fox jumps over the lazy dog while the river runs past the old stone mill and the wind moves through the tall grass. " * 140
c, n, dt = chat(para + "\n\nSummarize the passage above in one sentence.", 64)
print(f"[b: ~2000-token prompt] {n} tokens {dt:.1f}s garbled={chr(0xFFFD) in c or not c.strip()} :: {c[:90]!r}")
with cf.ThreadPoolExecutor(6) as ex:
    rs = list(ex.map(lambda i: chat(f"Write a 150-word explanation of topic {i}: why the sky is blue.", 160), range(6)))
bad = sum(1 for c, n, dt in rs if chr(0xFFFD) in c or not c.strip()); tot = sum(n for _, n, _ in rs)
print(f"[c: 6 streams] {tot} tokens, garbled streams {bad}/6 :: {rs[0][0][:70]!r}")
ok_d = gate("d: greedy gate again")
print("BATTERY", "PASS" if (ok_a and ok_d and bad == 0) else "FAIL")
