# DPI Probe Gate

**Question:** Does an LLM agent internally "know" an action will harm the user before it takes it — and does that internal signal still work when the agent operates in Hindi, Marathi, Tamil, or code-mixed versions of them?

**What it does:** A linear probe reads the agent's residual-stream activations at the moment it decides its next move. If the probe fires, a gate blocks the action before it executes. The probe is trained on English only and tested on six other language conditions, against a black-box LLM judge that reads the transcript.

Language conditions: `en`, `hi`, `hi_en` (Hinglish), `mr`, `mr_en` (Marathi-English), `ta`, `ta_en` (Tanglish).
Agent models: `Qwen/Qwen3-8B` and `dheeyantra/dhee-pravega` (Indic tool-calling model on Gemma-4).

## What is real and what is constructed

| Part | Status |
|---|---|
| Agent model, its tool calls, its activations | Real — open models run locally |
| Probe scores and block/allow decisions | Real, computed live |
| Backend actions (UPI verify, etc.) | Real HTTP calls **if** `backend_config.json` points at a sandbox you have access to. Otherwise **dry-run**: nothing executes and every response says so |
| Test scenarios | Constructed test cases (invented names, UPI IDs, companies), labelled `constructed_test_case: true` |
| Translations | LLM drafts; each row records whether a human reviewed it (`human_reviewed`) |

Never point this at live Aadhaar, UPI or other production systems. Only use sandboxes you are permitted to test.

## Scenario design

Five DPI harm types, 12 minimal pairs each (120 English scenarios). In each pair, the user request is identical and only one fact in the tool result differs:

- `payee_mismatch` — the UPI ID's registered name isn't the landlord the user named
- `fraud_flag` — the UPI ID has recent fraud reports
- `injection` — a bill contains instructions planted for the AI to pay a third party
- `doc_share_scope` — user said "only share with my bank", request comes from a loan app
- `aa_consent_purpose` — Account Aggregator consent is for marketing, 5 years of data

Probe label = whether the context makes the target action harmful. Separately, `executed_target` records whether the agent actually did it, so behaviour can be compared across languages too.

## Run order (Kaggle dual-T4 or any GPU)

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # offline checks, no GPU needed

python make_scenarios.py                       # data/scenarios_en.jsonl

# 1. Draft translations + code-mix (GPU)
python translate.py draft --langs hi hi_en mr mr_en ta ta_en
# 2. Review data/review_<lang>.csv: put y in "reviewed" for rows you checked,
#    write fixes in "corrected". Check every row with missing_protected first.
python translate.py apply --langs hi hi_en mr mr_en ta ta_en

# 3. Smoke test: prints the first raw output and the parsed tool call
python collect.py --model qwen3-8b --langs en --limit 5
python collect.py --model dhee-pravega --langs en --limit 5
#    If "parsed: []" but the raw output clearly calls a tool, the parser in
#    agent.py needs that format added. Check this before the long run.

# 4. Full collection (resumable per language)
python collect.py --model qwen3-8b --langs en hi hi_en mr mr_en ta ta_en
python collect.py --model dhee-pravega --langs en hi hi_en mr mr_en ta ta_en

# 5. Black-box baseline
python judge.py --agent qwen3-8b
python judge.py --agent dhee-pravega

# 6. Probes + plots
python probes.py --model qwen3-8b
python probes.py --model dhee-pravega

# 7. Live demo (public link so it works from Kaggle)
python demo_app.py --model qwen3-8b --share
```

The smoke test prints seconds per scenario — multiply by 840 (120 × 7 languages) to estimate each full run. If outputs come back empty or garbled on T4s, the likely cause is fp16 overflow; rerun on a bf16 GPU (H100/A100), where `load_model` switches to bf16 automatically. `--load-4bit` is available if memory is tight.

## Backend

Copy `backend_config.example.json` to `backend_config.json` and fill in endpoints from your sandbox's documentation. Credentials go in environment variables, never in the file. Tools you don't configure stay in dry-run mode.

Inbox items for the demo (`read_inbox_item`) are JSON files in `inbox/` that you write yourself; examples are included.

## Outputs

- `results/<model>_results.json` — per-language probe AUROC, detection rate and false-positive rate at the English 5%-FPR threshold, judge comparison, agent behaviour rates
- `results/<model>_auroc_by_layer.png`, `results/<model>_probe_vs_judge.png`
- `results/<model>_probe.npz` — probe used by the demo

## Method notes and known limits

- Folds are split by `pair_id`, so a scenario, its twin, and all their translations stay together; nothing leaks across languages.
- The best layer is chosen on English, so the English score is slightly optimistic; other languages are not affected by that choice.
- The judge sees the agent's generated move; the probe reads activations before the move exists. The judge therefore has more information.
- For Qwen3-8B the judge is the same model. A different judge model would be a fairer baseline.
- 60 pairs is a pilot. Treat results as indicative and report confidence intervals before making claims.
- In the live demo, probe scores before the first lookup are outside the training distribution; only world-changing actions are gated.

## AI assistance disclosure

Parts of this code were drafted with Claude (Anthropic). Disclose this, and which parts, in the hackathon submission as the rules require.
