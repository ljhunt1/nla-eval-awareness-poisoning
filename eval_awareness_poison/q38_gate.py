"""Gate for the zero-shot Control1-on-Qwen3.8 readout: FVE of Control1 (RL final) on the same 2,000 P2 positions, Qwen3.6
activations (q38/g36) vs Qwen3.8 activations (q38/g38). Proceed only if Qwen3.8's FVE is >= 0.25 and >= half of
Qwen3.6's; otherwise the NLA can't read Qwen3.8 and grading its explanations would be meaningless.
"""
import json
import sys

f36, f38 = (json.load(open(f"q38/{d}/args.json"))["fve"] for d in ("g36", "g38"))
ok = f38 >= 0.25 and f38 >= 0.5 * f36
json.dump(dict(fve36=f36, fve38=f38, ok=ok), open("q38/gate.json", "w"), indent=1)
print(f"gate: FVE Qwen3.6 {f36:.3f}, Qwen3.8 {f38:.3f} -> {'OK' if ok else 'STOP'}", flush=True)
sys.exit(0 if ok else 1)
