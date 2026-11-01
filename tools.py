"""Tool schemas the agent sees, plus backends that actually execute calls.

Backends:
  HttpSandboxBackend - sends real HTTP requests to a sandbox you have access to.
                       Configure endpoints in backend_config.json (see README).
  DryRunBackend      - executes nothing and says so in every response. Used when
                       no sandbox is configured, so nothing is ever faked.
  read_inbox_item is served from a local folder of items you write yourself
  (your own bills / requests), not from any external system.
"""

import json
import os
import string
from pathlib import Path

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "verify_payee",
            "description": "Look up a UPI ID (VPA) and return the registered account holder name and any risk information.",
            "parameters": {
                "type": "object",
                "properties": {"vpa": {"type": "string", "description": "UPI ID, e.g. name@okbank"}},
                "required": ["vpa"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox_item",
            "description": "Read a bill, document request or consent request from the user's inbox.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_upi_payment",
            "description": "Send money over UPI to a VPA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vpa": {"type": "string"},
                    "amount_inr": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["vpa", "amount_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_document",
            "description": "Share a document from the user's DigiLocker with a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_type": {"type": "string", "description": "e.g. aadhaar, pan, salary_slip"},
                    "recipient": {"type": "string"},
                },
                "required": ["doc_type", "recipient"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grant_aa_consent",
            "description": "Approve an Account Aggregator consent request so a financial institution can fetch the user's data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consent_id": {"type": "string"},
                    "fiu_name": {"type": "string"},
                },
                "required": ["consent_id", "fiu_name"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


class DryRunBackend:
    """Executes nothing. Every response states that plainly."""

    name = "dry-run"
    is_real = False

    def execute(self, tool, args):
        return {"executed": False, "backend": "dry-run",
                "message": f"No sandbox configured; {tool} was NOT executed."}


class HttpSandboxBackend:
    """Sends real requests to a sandbox described in a JSON config.

    Config format (one entry per tool you have a sandbox endpoint for):
    {
      "headers": {"x-client-id": "$SANDBOX_CLIENT_ID", "x-client-secret": "$SANDBOX_SECRET"},
      "tools": {
        "verify_payee": {"method": "POST", "url": "https://sandbox.example.com/upi/verify",
                          "body": {"vpa": "$vpa"}}
      }
    }
    "$NAME" in headers reads an environment variable; "$arg" in body/url fills
    in the tool argument of that name. Tools missing from the config fall back
    to dry-run so nothing is ever silently simulated.
    """

    name = "http-sandbox"
    is_real = True

    def __init__(self, config_path):
        import requests  # local import so offline tests don't need it

        self._requests = requests
        self.cfg = json.loads(Path(config_path).read_text())
        self.fallback = DryRunBackend()

    def _headers(self):
        out = {}
        for k, v in self.cfg.get("headers", {}).items():
            if isinstance(v, str) and v.startswith("$"):
                v = os.environ.get(v[1:], "")
            out[k] = v
        return out

    @staticmethod
    def _fill(obj, args):
        if isinstance(obj, str):
            if obj.startswith("$") and obj[1:] in args:
                return args[obj[1:]]
            return string.Template(obj).safe_substitute({k: str(v) for k, v in args.items()})
        if isinstance(obj, dict):
            return {k: HttpSandboxBackend._fill(v, args) for k, v in obj.items()}
        if isinstance(obj, list):
            return [HttpSandboxBackend._fill(v, args) for v in obj]
        return obj

    def execute(self, tool, args):
        spec = self.cfg.get("tools", {}).get(tool)
        if spec is None:
            return self.fallback.execute(tool, args)
        url = self._fill(spec["url"], args)
        body = self._fill(spec.get("body"), args)
        try:
            r = self._requests.request(spec.get("method", "POST"), url, json=body,
                                       headers=self._headers(), timeout=20)
            try:
                payload = r.json()
            except ValueError:
                payload = {"raw": r.text[:2000]}
            return {"executed": True, "backend": "http-sandbox", "status_code": r.status_code,
                    "response": payload}
        except Exception as e:  # network errors are reported, never hidden
            return {"executed": False, "backend": "http-sandbox", "error": repr(e)}


class LocalInbox:
    """Serves read_inbox_item from JSON files you write in a folder."""

    def __init__(self, folder="inbox"):
        self.folder = Path(folder)

    def read(self, item_id):
        p = self.folder / f"{item_id}.json"
        if not p.exists():
            return {"error": f"no inbox item {item_id}"}
        return json.loads(p.read_text(encoding="utf-8"))


def make_backend(config_path=None):
    if config_path and Path(config_path).exists():
        return HttpSandboxBackend(config_path)
    return DryRunBackend()
