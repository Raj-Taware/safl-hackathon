"""Live demo: a real model acts as the agent, the probe reads its real
activations at every step, and the gate blocks flagged actions before they
reach the backend.

python demo_app.py --model qwen3-8b --backend backend_config.json --share
python demo_app.py --model qwen3-8b --cli "Pay my electricity bill, item BILL-1234"

What is real: the model, its activations, the probe score, the block/allow
decision, and (if backend_config.json is set up) the sandbox HTTP calls.
Without a backend config, actions run in dry-run mode and the UI says so.
Inbox items come from JSON files you write in ./inbox.
"""

import argparse
import json

import numpy as np

from agent import load_model, render_prompt, generate_with_acts, parse_tool_calls
from config import MODELS, SYSTEM_PROMPT, ACTION_TOOLS
from tools import LocalInbox, make_backend

MAX_STEPS = 4


class Probe:
    def __init__(self, path):
        z = np.load(path)
        self.layer = int(z["layer"])
        self.mu, self.sd, self.w = z["mu"], z["sd"], z["w"]
        self.b, self.threshold = float(z["b"]), float(z["threshold"])

    def score(self, acts):
        x = acts[self.layer].astype(np.float32)
        return float(((x - self.mu) / self.sd) @ self.w + self.b)


def gate_decision(tool_name, score, threshold):
    """Only world-changing actions are gated; lookups always run."""
    if tool_name in ACTION_TOOLS and score > threshold:
        return "BLOCK"
    return "ALLOW"


def run_agent(user_msg, tok, model, probe, backend, inbox):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]
    trace = []
    for step in range(MAX_STEPS):
        text, acts = generate_with_acts(tok, model, render_prompt(tok, messages))
        s = probe.score(acts)
        calls = parse_tool_calls(text)
        if not calls:
            trace.append({"step": step, "probe_score": s, "reply": text, "calls": []})
            break
        entry = {"step": step, "probe_score": s, "calls": [], "raw": text}
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls]})
        stop = False
        for c in calls:
            decision = gate_decision(c["name"], s, probe.threshold)
            if decision == "BLOCK":
                result = {"blocked": True, "reason": "Safety monitor flagged this action before execution."}
                stop = True
            elif c["name"] == "read_inbox_item":
                result = inbox.read(c["arguments"].get("item_id", ""))
            else:
                result = backend.execute(c["name"], c["arguments"])
            entry["calls"].append({"call": c, "decision": decision, "result": result})
            messages.append({"role": "tool", "name": c["name"],
                             "content": json.dumps(result, ensure_ascii=False)})
        trace.append(entry)
        if stop:
            break
    return trace


def trace_markdown(trace, probe, backend):
    banner = ("**Backend: real sandbox HTTP calls**" if backend.is_real
              else "**Backend: DRY RUN — no action is executed anywhere**")
    lines = [banner, f"Probe layer {probe.layer}, block threshold {probe.threshold:.2f}", ""]
    for e in trace:
        lines.append(f"### Step {e['step']}  ·  probe score {e['probe_score']:.2f}")
        if not e["calls"]:
            lines.append(e.get("reply", ""))
        for c in e["calls"]:
            mark = "🛑 BLOCKED" if c["decision"] == "BLOCK" else "✅ allowed"
            lines.append(f"- `{c['call']['name']}({json.dumps(c['call']['arguments'], ensure_ascii=False)})` → {mark}")
            lines.append(f"  - result: `{json.dumps(c['result'], ensure_ascii=False)[:400]}`")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(MODELS))
    ap.add_argument("--probe", default=None, help="defaults to results/<model>_probe.npz")
    ap.add_argument("--backend", default="backend_config.json")
    ap.add_argument("--inbox", default="inbox")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--cli", default=None, help="run one request and print the trace")
    ap.add_argument("--share", action="store_true", help="public Gradio link (needed on Kaggle)")
    a = ap.parse_args()

    probe = Probe(a.probe or f"results/{a.model}_probe.npz")
    backend, inbox = make_backend(a.backend), LocalInbox(a.inbox)
    tok, model = load_model(MODELS[a.model], a.load_4bit)

    if a.cli:
        print(trace_markdown(run_agent(a.cli, tok, model, probe, backend, inbox), probe, backend))
        return

    import gradio as gr

    def respond(msg):
        return trace_markdown(run_agent(msg, tok, model, probe, backend, inbox), probe, backend)

    with gr.Blocks(title="DPI agent safety gate") as ui:
        gr.Markdown("## Activation-probe gate for a DPI agent\nType a request in English, Hindi, "
                    "Marathi, Tamil or a code-mixed version. The probe reads the model's internal "
                    "activations at each step; flagged actions are blocked before they execute.")
        box = gr.Textbox(label="Your request", lines=2)
        out = gr.Markdown()
        gr.Button("Run agent").click(respond, box, out)
    ui.launch(share=a.share)


if __name__ == "__main__":
    main()
