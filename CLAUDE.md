# Handoff: DPI Probe Gate

Read this first, then README.md. This file gives the context, the decisions behind them, the status, and the next steps.

## Who and why

- Owner: Raj, final-year EEE + Data Science student at BITS Pilani Goa, working solo.
- Interest: pragmatic mechanistic interpretability (in the sense of the GDM "pragmatic interpretability" agenda: proxy tasks, beat black-box baselines).
- Uses of this project:
  1. Entry for the SAFL **India Agentic AI Security Hackathon**. Deadline **Sat 31 Oct 2026, 11:59 PM IST**. One entry, solo is allowed. Submission form: bit.ly/saflhacksubmission.
  2. Evidence for an application to the Secure AI Futures Lab (SAFL) Research Fellow internship track.
  3. A workshop paper and a blog post after about 2 months.

## How Raj wants you to work

- Ask clarifying questions where a decision is his, and wait for his answer before you continue.
- If you don't know something, say so. Do not guess, speculate or fabricate data, results or citations.
- Never fake or simulate results, backend responses or a working demo. If something has not run, say it has not run.

## The research question

Does an LLM agent internally represent that an action will harm the user, before it takes the action? Does that signal survive when the agent operates in Hindi, Marathi, Tamil and their code-mixed versions?

- Probe: linear probe on residual-stream activations at the **decision point** (last prompt token after the lookup result, before the agent writes its next move).
- Train on English only. Test on `hi`, `hi_en`, `mr`, `mr_en`, `ta`, `ta_en`.
- Baseline: black-box LLM judge that reads the transcript and the agent's move.
- Demo: a gate that blocks a world-changing tool call when the probe score passes the threshold.

## Decisions already made (do not reopen without asking Raj)

| Decision | Reason |
|---|---|
| Agent models: `Qwen/Qwen3-8B` and `dheeyantra/dhee-pravega` (Gemma-4 E2B, Indic tool calling) | Two models show whether findings generalise; Dhee is built for Indic tool calls |
| Seven language conditions above | Raj chose Marathi, Hindi, Tamil + code-mixed. He checks hi, hi_en, mr, mr_en; friends check ta, ta_en |
| Code-mix made as LLM drafts, then human-corrected | Translation models output pure script only |
| 5 harm domains × 12 minimal pairs = 120 English scenarios | Minimal pairs stop probes from scoring on surface differences |
| Folds split by `pair_id` across all languages | Prevents translated copies of training items leaking into test |
| Threshold fixed at 5% FPR on English, applied unchanged to other languages | Mirrors how a deployed gate behaves |
| Judge: `sarvamai/sarvam-m` for both agents (4-bit on T4) | Raj chose it 27 Oct: covers hi/mr/ta, not gated, different family from Qwen and Gemma |
| Backend: swappable. Real HTTP sandbox if configured, otherwise explicit dry-run | Raj will say which sandbox. No faked responses ever |

## Hard constraints

- Hackathon rules: test **only** in provided DPI sandboxes or systems Raj owns or has permission to test. **Never** call live Aadhaar, UPI or other production systems. Do not publish exploits that put real users at risk.
- Rules require disclosure of AI tool use. README has a disclosure section; keep it accurate.
- Code written before the event must be cited; the core work must happen in the hackathon window (it does, from 27 Oct).
- Compute: Kaggle dual-T4, Colab free, occasional BITS HPC H100, vast.ai. About 7-9B open models, inference and activation access only.

## Status (as of 27 Oct 2026, evening IST)

Verified:
- `python -m pytest tests -q` passes (12 tests, offline, no GPU).
- Both models' real chat templates render the scenario prompts (checked with downloaded tokenizers).
- Dhee-Pravega emits `<|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>`; `agent.parse_tool_calls` handles it (tested on a hand-written sample, not real model output).
- `data/scenarios_en.jsonl` generated (120 rows, seed 0).

Not verified (nothing has run on a GPU yet):
- Model loading, generation, activation capture, tool-call parsing on real outputs, speed per scenario.
- fp16 behaviour on T4s. `load_model` uses bf16 when the GPU supports it, else fp16. Empty or garbled output on T4 likely means fp16 overflow; move to a bf16 GPU.
- Translation quality; review sheets do not exist yet.
- Any backend. `backend_config.json` does not exist; everything runs dry-run.

## File map

- `config.py` — languages, model ids, action vs lookup tools, system prompt
- `tools.py` — tool schemas; `HttpSandboxBackend`, `DryRunBackend`, `LocalInbox`
- `make_scenarios.py` — paired English scenarios
- `translate.py` — `draft` (LLM translations + review CSVs) and `apply` (merge human fixes)
- `agent.py` — model loading, prompt building, generation + decision-point activations, tool-call parsing, `executed_target`
- `collect.py` — run agent over scenarios, save `data/<model>/<lang>_acts.npy` and `_meta.jsonl` (resumable)
- `judge.py` — black-box judge scores
- `probes.py` — cross-validated per-layer probes, metrics, plots, final probe for the demo
- `demo_app.py` — Gradio live gate (`--share` for Kaggle), or `--cli`
- `inbox/` — example inbox items Raj wrote (user-owned content for `read_inbox_item`)
- `tests/test_offline.py` — offline tests

## Next steps, in order

1. Smoke test on GPU: `python collect.py --model qwen3-8b --langs en --limit 5`, same for `dhee-pravega`. Check the printed raw output parses into tool calls. If it doesn't, extend `parse_tool_calls` and add a test with the real output. Report seconds per scenario × 840 to Raj so he can judge whether full runs fit before Saturday.
2. `python translate.py draft --langs hi hi_en mr mr_en ta ta_en`. Send Raj the review CSVs. Rows with `missing_protected` need attention first.
3. After Raj and friends review: `python translate.py apply ...`.
4. Full `collect.py` runs for both models, then `judge.py`, then `probes.py`.
5. When Raj gives sandbox details: write `backend_config.json` from the sandbox's docs (credentials in env vars only), and test one call. Check the sandbox is allowed under the hackathon rules.
6. Run the demo with real requests in several languages; record a short video for the submission.
7. Submission write-up: threat model, method, results with honest sample sizes, limitations, adoption path (IndiaAI Safety Institute, RBI, MeitY), AI-use disclosure.

If time runs short before Saturday, ask Raj before cutting scope. Possible cuts: one model instead of two, or fewer languages in the hackathon version, with the rest in the 2-month version.

## Open questions for Raj

- Which sandbox backend (SAFL-provided, a vendor sandbox such as Cashfree's verification sandbox, or other)? He said he will confirm.
- What the SAFL submission form asks for (video, repo, write-up). Not yet known.

## Known method limits (keep these visible in any write-up)

- 60 pairs is a pilot. Report confidence intervals; don't overclaim.
- Best layer is picked on English, so the English score is slightly optimistic.
- The judge sees the generated move; the probe reads activations before the move exists.
- In the live demo, scores before the first lookup are outside the training distribution; only action tools are gated.

## Related prior work (from the novelty check)

DeceptGuard/DeceptArena (black-box vs CoT vs activation-probe monitors in agentic settings); black-to-white performance boost benchmark for deception probes; PRISMS (neuron detectors for tool-use failures); "Before Retrieval Begins" (harmful tool admission, Oct 2026; whether it uses activations was not confirmed); IndicSafe and CompositeHarm (uneven safety across Indic languages); cross-lingual shared safety neurons (ICML 2026). No exact match to "activation gate for harmful DPI agent actions, tested across Indic and code-mixed languages" was found, but the search was not exhaustive. Re-check before writing the paper.
