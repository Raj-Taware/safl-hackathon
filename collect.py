"""Run the agent on every scenario and save decision-point activations.

python collect.py --model qwen3-8b --langs en hi hi_en mr mr_en ta ta_en
python collect.py --model dhee-pravega --langs en hi hi_en mr mr_en ta ta_en

Writes, per model and language:
  data/<model>/<lang>_acts.npy    float16 [n_scenarios, n_layers+1, d_model]
  data/<model>/<lang>_meta.jsonl  id, pair_id, harmful, raw output, parsed calls,
                                  executed_target (did it actually do the action)
A language is skipped if its files already exist, so a crashed Kaggle
session can resume.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agent import (load_model, scenario_messages, render_prompt, generate_with_acts,
                   parse_tool_calls, executed_target)
from config import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N scenarios only (smoke test)")
    a = ap.parse_args()

    outdir = Path("data") / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(MODELS[a.model], a.load_4bit)

    for lang in a.langs:
        # Smoke runs (--limit) write separate files, so they never make a full run skip this language.
        stem = f"{lang}_smoke" if a.limit else lang
        acts_p, meta_p = outdir / f"{stem}_acts.npy", outdir / f"{stem}_meta.jsonl"
        if acts_p.exists() and meta_p.exists() and not a.limit:
            print(f"skip {lang} (done)")
            continue
        scen = [json.loads(l) for l in open(f"data/scenarios_{lang}.jsonl", encoding="utf-8")]
        if a.limit:
            scen = scen[:a.limit]
        all_acts, t0 = [], time.time()
        with open(meta_p, "w", encoding="utf-8") as mf:
            for i, sc in enumerate(scen):
                prompt = render_prompt(tok, scenario_messages(sc))
                text, acts = generate_with_acts(tok, model, prompt)
                calls = parse_tool_calls(text)
                all_acts.append(acts)
                mf.write(json.dumps({
                    "id": sc["id"], "pair_id": sc["pair_id"], "domain": sc["domain"],
                    "lang": lang, "harmful": sc["harmful"],
                    "human_reviewed": sc.get("human_reviewed", lang == "en"),
                    "output": text, "calls": calls,
                    "executed_target": executed_target(calls, sc),
                }, ensure_ascii=False) + "\n")
                if i == 0:
                    print("---- first raw output (check the tool-call format parses) ----")
                    print(text[:600])
                    print("parsed:", calls)
                if (i + 1) % 20 == 0:
                    rate = (time.time() - t0) / (i + 1)
                    print(f"{lang}: {i+1}/{len(scen)}  {rate:.1f}s each")
        np.save(acts_p, np.stack(all_acts))
        rate = (time.time() - t0) / len(scen)
        print(f"saved {lang}: {len(scen)} scenarios, {rate:.1f}s each "
              f"(x840 = {rate * 840 / 3600:.1f} h for a full run)")


if __name__ == "__main__":
    main()
