#!/usr/bin/env python3
"""
test_pipeline.py: End-to-End Pipeline Verification Script
Jaypee Agentic AI Summit 2026 — Track 2 | Healthcare

1. Loads a sample dermoscopic image (or synthesizes one if needed).
2. Runs Agents 1 to 6 in order through orchestrator.py.
3. Prints each agent's output, final decision (AUTO_TRIAGE / REFER / RE_IMAGE),
   and the audit chain string.
4. Simulates a POST to /api/analyze and /api/triage using handler in api/index.py,
   confirming 200 responses with valid JSON.
"""

import os
import sys
import io
import json
import base64
import urllib.request
from http.server import HTTPServer
import threading
import time
from PIL import Image, ImageDraw

# Add workspace and api directories to path
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
api_dir = os.path.join(this_dir, "api")
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from orchestrator import agentic_dermaguard
from api.index import handler

def get_or_create_sample_image() -> Image.Image:
    """Finds an existing sample dermoscopic image or generates a realistic synthetic lesion."""
    candidates = [
        os.path.join(this_dir, "Health", "eval", "images", "ISIC_0024341.jpg"),
        os.path.join(this_dir, "Health", "eval", "images", "ISIC_0024332.jpg")
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[Image Source] Loaded existing image from {c}")
            return Image.open(c).convert("RGB")

    print("[Image Source] Synthesizing dermoscopic test image...")
    # Synthetic realistic dermoscopy image (pigmented lesion on skin)
    img = Image.new("RGB", (224, 224), (218, 178, 155))
    draw = ImageDraw.Draw(img)
    # Lesion outer perimeter
    draw.ellipse([50, 50, 174, 174], fill=(110, 65, 45), outline=(90, 50, 35), width=2)
    # Lesion central pigment network
    draw.ellipse([70, 75, 150, 155], fill=(65, 35, 25))
    draw.ellipse([90, 90, 130, 130], fill=(40, 20, 15))
    return img

def run_pipeline_test():
    print("=" * 70)
    print("  AGENTIC DERMAGUARD: MULTI-AGENT PIPELINE TEST (AGENTS 1 - 6)")
    print("=" * 70)

    # 1. Load or synthesize test image
    pil_image = get_or_create_sample_image()

    # 2. Run Agents 1 to 6 in order through orchestrator
    print("\n[Step 2] Executing Agents 1 - 6 through orchestrator.py ...")
    t0 = time.time()
    res = agentic_dermaguard.process_image(
        pil_image,
        patient_context={"age": 42, "sex": "female", "localization": "lower extremity"}
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    # 3. Print each agent's output and final decision
    print("\n" + "-" * 70)
    print("INDIVIDUAL AGENT AUDIT OUTPUTS:")
    print("-" * 70)
    agents = res["agents"]

    # Agent 1
    a1 = agents["A1_quality"]
    print(f"Agent 1 (Image Quality Gate):")
    print(f"  Status: {a1.get('status')} | Score: {a1.get('score')}/100")
    print(f"  Summary: {a1.get('summary')}")
    print(f"  Evidence: {len(a1.get('evidence', []))} items verified")

    # Agent 2
    a2 = agents["A2_diagnosis"]
    print(f"\nAgent 2 (Pathology Diagnosis Classifier):")
    print(f"  Predicted Class: {a2.get('predicted_code', a2.get('status'))} ({a2.get('predicted_class')})")
    print(f"  Confidence: {a2.get('confidence', a2.get('score'))*100:.2f}% | Top-2 Margin: {a2.get('margin_to_next', 0.0):.3f}")
    print(f"  Summary: {a2.get('summary')}")

    # Agent 3
    a3 = agents["A3_lesion_analysis"]
    print(f"\nAgent 3 (Lesion ABCD Morphometry & Edge Mapping):")
    print(f"  Status: {a3.get('status')} | Suspicion Score: {a3.get('suspicion_score', a3.get('score')):.2f}")
    print(f"  TDS Score: {a3.get('tds_score', 'N/A')}")
    print(f"  Summary: {a3.get('summary')}")

    # Agent 4
    a4 = agents["A4_context_reasoning"]
    print(f"\nAgent 4 (Epidemiological & Demographic Reasoning):")
    print(f"  Risk Indicator: {a4.get('risk_indicator', a4.get('status'))} | Score: {a4.get('score')}")
    print(f"  Summary: {a4.get('summary')}")

    # Agent 5
    a5 = agents["A5_safety"]
    print(f"\nAgent 5 (Clinical Safety Sentinel & Veto):")
    print(f"  Safety Flag: {a5.get('safety_flag', a5.get('status'))} | Veto Auto-Triage: {a5.get('veto_auto_triage', False)}")
    print(f"  Summary: {a5.get('summary')}")

    # Agent 6
    a6 = agents["A6_coordinator"]
    print(f"\nAgent 6 (Coordinator & Master Decision Policy):")
    print(f"  Disposition: {a6.get('final_decision', a6.get('status'))}")
    print(f"  Policy Rule: {a6.get('policy_rule')}")
    print(f"  Summary: {a6.get('summary')}")

    print("\n" + "=" * 70)
    print(f"FINAL DECISION: {res['final_decision']}")
    print(f"PREDICTED CODE: {res['prediction']} ({res.get('prediction_name', '')})")
    print(f"TRACEABLE CHAIN:\n  {res['traceable_chain']}")
    print(f"REQUIRED UI EXAMPLE STRIP:\n  {res['example_strip']}")
    print(f"PIPELINE LATENCY: {elapsed_ms:.1f} ms")
    print("=" * 70)

    # Validations
    assert res['prediction'] in {'mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'}, f"Invalid prediction: {res['prediction']}"
    assert res['final_decision'] in {'AUTO_TRIAGE', 'REFER', 'RE_IMAGE'}, f"Invalid decision: {res['final_decision']}"
    assert "example_strip" in res and " | " in res["example_strip"], "Missing example_strip"

    # 4. Simulate POST to /api/analyze and /api/triage using handler in api/index.py
    print("\n[Step 4] Starting in-process HTTP test server to verify /api/analyze and /api/triage ...")
    port = 8899
    server = HTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        # Convert test image to base64
        buf = io.BytesIO()
        pil_image.resize((224, 224)).save(buf, format="JPEG", quality=85)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        post_body = json.dumps({
            "image": img_b64,
            "context": {"age": 42, "sex": "female", "localization": "lower extremity"}
        }).encode("utf-8")

        # Test POST /api/analyze
        req1 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/analyze",
            data=post_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req1) as resp1:
            assert resp1.status == 200, f"/api/analyze returned status {resp1.status}"
            data1 = json.loads(resp1.read().decode("utf-8"))
            assert "prediction" in data1 and "decision" in data1, "Missing prediction or decision in /api/analyze"
            print(f"[PASS] POST /api/analyze -> HTTP {resp1.status} OK")
            print(f"       Prediction: {data1['prediction']} | Decision: {data1['decision']}")

        # Test POST /api/triage
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/triage",
            data=post_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req2) as resp2:
            assert resp2.status == 200, f"/api/triage returned status {resp2.status}"
            data2 = json.loads(resp2.read().decode("utf-8"))
            assert "prediction" in data2 and "decision" in data2, "Missing prediction or decision in /api/triage"
            print(f"[PASS] POST /api/triage -> HTTP {resp2.status} OK")
            print(f"       Prediction: {data2['prediction']} | Decision: {data2['decision']}")

    finally:
        server.shutdown()

    print("\n" + "=" * 70)
    print("  [SUCCESS] All pipeline checks, agent contracts, and API routes passed 100%!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline_test()
