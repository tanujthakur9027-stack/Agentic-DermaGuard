#!/usr/bin/env python3
"""
scripts/submit.py
Submits the final submission CSV to the organizers' API.
Jaypee Agentic AI International Summit 2026 - Track 2: Healthcare

Features:
- Configurable via config/submit.json or CLI flags / environment variables
- Refuses to submit unless the validator passes
- Supports --dry-run to inspect exact HTTP request with masked auth tokens
- Interactive confirmation showing row count and decision breakdown
- Supports 3 payload modes: "multipart_file", "json_rows", "raw_csv"
- Retries on network errors with backoff
- Records response to outputs/submit_response.json
"""

import os
import sys
import json
import argparse
import time
import urllib.request
import urllib.error
import mimetypes
import uuid
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_submission import validate_submission

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "submit.json")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = {
        "url": os.environ.get("DERMAGUARD_API_URL", ""),
        "http_method": os.environ.get("DERMAGUARD_HTTP_METHOD", "POST"),
        "auth_header_name": os.environ.get("DERMAGUARD_AUTH_HEADER", "Authorization"),
        "auth_token": os.environ.get("DERMAGUARD_AUTH_TOKEN", ""),
        "payload_mode": os.environ.get("DERMAGUARD_PAYLOAD_MODE", "multipart_file"),
        "file_field_name": os.environ.get("DERMAGUARD_FILE_FIELD", "file"),
        "json_key_predictions": os.environ.get("DERMAGUARD_JSON_KEY", "predictions"),
        "extra_fields": {}
    }

    path_to_try = config_path or DEFAULT_CONFIG_PATH
    if os.path.exists(path_to_try):
        try:
            with open(path_to_try, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                cfg.update(file_cfg)
        except Exception as e:
            print(f"[WARN] Failed to read config from {path_to_try}: {e}")

    return cfg

def mask_token(token: str) -> str:
    if not token:
        return "<none>"
    if len(token) <= 8:
        return "*" * len(token)
    return token[:3] + "*" * (len(token) - 6) + token[-3:]

def build_multipart_body(fields: Dict[str, str], filepath: str, field_name: str = "file") -> Tuple[bytes, str]:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{v}\r\n".encode("utf-8"))

    filename = os.path.basename(filepath)
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: text/csv\r\n\r\n")
    with open(filepath, "rb") as f:
        body.extend(f.read())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type

def prepare_request(cfg: Dict[str, Any], csv_path: str) -> Tuple[str, str, Dict[str, str], bytes, str]:
    url = cfg.get("url", "").strip()
    method = cfg.get("http_method", "POST").upper()
    headers: Dict[str, str] = {}
    auth_header = cfg.get("auth_header_name", "Authorization").strip()
    auth_token = cfg.get("auth_token", "").strip()

    if auth_token:
        if auth_header.lower() == "authorization" and not auth_token.startswith("Bearer ") and not auth_token.startswith("Basic "):
            headers[auth_header] = f"Bearer {auth_token}"
        else:
            headers[auth_header] = auth_token

    payload_mode = cfg.get("payload_mode", "multipart_file")
    extra_fields = cfg.get("extra_fields", {})

    if payload_mode == "multipart_file":
        field_name = cfg.get("file_field_name", "file")
        body_bytes, ct = build_multipart_body(extra_fields, csv_path, field_name=field_name)
        headers["Content-Type"] = ct
        preview = f"[Multipart Form-Data: {field_name}=@{os.path.basename(csv_path)}, {len(body_bytes)} bytes]"
    elif payload_mode == "raw_csv":
        with open(csv_path, "rb") as f:
            body_bytes = f.read()
        headers["Content-Type"] = "text/csv"
        preview = body_bytes[:300].decode("utf-8", errors="replace") + "..."
    elif payload_mode == "json_rows":
        df = pd.read_csv(csv_path, dtype=str)
        rows_list = []
        for _, r in df.iterrows():
            rows_list.append({
                "image_id": str(r["image_id"]).strip(),
                "prediction": str(r["prediction"]).strip(),
                "decision": str(r["decision"]).strip()
            })
        json_key = cfg.get("json_key_predictions", "predictions")
        payload_dict = {json_key: rows_list}
        payload_dict.update(extra_fields)
        body_str = json.dumps(payload_dict)
        body_bytes = body_str.encode("utf-8")
        headers["Content-Type"] = "application/json"
        preview = body_str[:300] + "..."
    else:
        raise ValueError(f"Unknown payload_mode: {payload_mode}")

    return url, method, headers, body_bytes, preview

def execute_submission(
    csv_path: str,
    ids_path: Optional[str] = None,
    config_path: Optional[str] = None,
    dry_run: bool = False,
    yes: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    print("=" * 65)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare | Submission Client")
    print("=" * 65)

    # 1. Validation check FIRST - refuse to submit if invalid!
    print(f"[CHECK] Validating submission CSV: {csv_path}...")
    passed, errs, stats = validate_submission(csv_path, ids_path=ids_path)
    if not passed:
        print("\n[REFUSAL] SUBMISSION BLOCKED! The file failed validation:", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return False, {"error": "Validation failed", "validation_errors": errs}

    print("[PASS] Submission CSV passed all integrity and schema checks.")
    print("-" * 65)
    print("CONFIRMATION SUMMARY:")
    print(f"  Total Rows:         {stats['total_rows']}")
    for d, c in stats["decisions"].items():
        print(f"  {d:<18}: {c}")
    print("-" * 65)

    # 2. Load Config
    cfg = load_config(config_path)
    url = cfg.get("url", "").strip()
    if not url:
        print("[ERROR] API URL is not configured in config/submit.json or DERMAGUARD_API_URL", file=sys.stderr)
        return False, {"error": "Missing API URL"}

    # 3. Prepare Request
    url, method, headers, body_bytes, preview = prepare_request(cfg, csv_path)

    # Mask headers for display
    display_headers = dict(headers)
    auth_header_name = cfg.get("auth_header_name", "Authorization")
    if auth_header_name in display_headers:
        display_headers[auth_header_name] = mask_token(display_headers[auth_header_name])

    print("HTTP REQUEST SPECIFICATION:")
    print(f"  Endpoint:     {url}")
    print(f"  HTTP Method:  {method}")
    print(f"  Headers:      {json.dumps(display_headers, indent=4)}")
    print(f"  Payload Mode: {cfg.get('payload_mode')}")
    print(f"  Body Preview: {preview}")
    print(f"  Total Size:   {len(body_bytes)} bytes")
    print("-" * 65)

    if dry_run:
        print("[DRY-RUN] Execution halted successfully (--dry-run specified). No network request sent.")
        return True, {
            "dry_run": True,
            "url": url,
            "method": method,
            "headers": display_headers,
            "payload_preview": preview,
            "status": "DRY_RUN_OK"
        }

    # 4. Confirmation check
    if not yes:
        try:
            user_input = input(f"Proceed with official submission of {stats['total_rows']} rows to '{url}'? (y/N): ").strip()
            if user_input.lower() not in {"y", "yes"}:
                print("[ABORTED] User aborted submission.")
                return False, {"error": "User cancelled"}
        except EOFError:
            pass

    # 5. Send with retry and backoff
    print(f"[TRANSMIT] Sending submission to {url}...")
    max_retries = 3
    resp_obj = {}

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_code = resp.getcode()
                resp_body = resp.read().decode("utf-8", errors="replace")
                print(f"[SUCCESS] Submission received by server! HTTP {resp_code}")
                print(f"Response Body:\n{resp_body}")
                resp_obj = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status_code": resp_code,
                    "response": resp_body,
                    "submission_file": csv_path,
                    "total_rows": stats['total_rows']
                }
                break
        except urllib.error.HTTPError as e:
            resp_code = e.code
            resp_body = e.read().decode("utf-8", errors="replace")
            print(f"[HTTP ERROR] Server returned HTTP {resp_code}:\n{resp_body}", file=sys.stderr)
            resp_obj = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status_code": resp_code,
                "error": resp_body
            }
            break
        except Exception as ex:
            print(f"[NETWORK ERROR] Attempt {attempt}/{max_retries} failed: {ex}", file=sys.stderr)
            if attempt < max_retries:
                wait_s = 2 ** attempt
                print(f"Retrying in {wait_s} seconds...")
                time.sleep(wait_s)
            else:
                resp_obj = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status_code": 0,
                    "error": str(ex)
                }

    # Save outputs/submit_response.json
    out_resp_file = os.path.join(PROJECT_ROOT, "outputs", "submit_response.json")
    os.makedirs(os.path.dirname(out_resp_file), exist_ok=True)
    with open(out_resp_file, "w", encoding="utf-8") as f:
        json.dump(resp_obj, f, indent=2)
    print(f"[OUTPUT] Saved submission response log to: {out_resp_file}")

    return resp_obj.get("status_code", 0) in {200, 201, 202}, resp_obj

def main():
    parser = argparse.ArgumentParser(description="Agentic DermaGuard API Submitter")
    parser.add_argument("--csv", type=str, default="outputs/submission.csv", help="Submission CSV path")
    parser.add_argument("--ids", type=str, default=None, help="Sample IDs CSV reference to validate against")
    parser.add_argument("--config", type=str, default=None, help="Path to submit config JSON (default config/submit.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print request details without transmitting")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    success, _ = execute_submission(
        csv_path=args.csv,
        ids_path=args.ids,
        config_path=args.config,
        dry_run=args.dry_run,
        yes=args.yes
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
