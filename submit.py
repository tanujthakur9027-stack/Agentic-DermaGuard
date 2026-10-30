import os
import sys
import json
import argparse
import urllib.request
import urllib.error

def submit_csv_to_api(csv_path: str = "outputs/submission.csv", api_url: str = None, team_name: str = None, team_token: str = None):
    """
    Submits the official Level 2 CSV file (image_id, prediction, decision) to the organizers' API.
    """
    if not os.path.exists(csv_path):
        print(f"[Error] Submission CSV '{csv_path}' does not exist! Run evaluate.py first.")
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    lines = [l.strip() for l in csv_content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        print("[Error] CSV file appears empty or missing rows.")
        return False

    header = lines[0]
    expected_header = "image_id,prediction,decision"
    if header.replace(" ", "") != expected_header:
        print(f"[Warning] CSV header '{header}' differs from expected '{expected_header}'.")

    print("=" * 65)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare — Level 2 Hidden-Test CSV Submission Client")
    print("=" * 65)
    print(f"Total rows to submit: {len(lines) - 1}")
    print(f"First 3 rows:\n" + "\n".join(lines[:4]))
    print("=" * 65)

    if not api_url:
        api_url = os.environ.get("DERMAGUARD_API_URL")
    if not api_url:
        api_url = input("Enter the Summit CSV Submission API URL: ").strip()

    if not team_name:
        team_name = os.environ.get("DERMAGUARD_TEAM_NAME", "Team-DermaGuard")
        user_team = input(f"Enter Team Name [{team_name}]: ").strip()
        if user_team:
            team_name = user_team

    if not team_token:
        team_token = os.environ.get("DERMAGUARD_TEAM_TOKEN", "")
        if not team_token:
            team_token = input("Enter Team Token / API Key (if provided, else leave blank): ").strip()

    # Determine if API expects raw CSV or JSON with embedded rows
    rows_list = []
    for l in lines[1:]:
        parts = l.split(",")
        if len(parts) >= 3:
            rows_list.append({
                "image_id": parts[0].strip(),
                "prediction": parts[1].strip(),
                "decision": parts[2].strip()
            })

    payload = {
        "team_name": team_name,
        "track": "Track 2 | Healthcare",
        "level": 2,
        "total_test_images": len(rows_list),
        "csv_raw": csv_content,
        "rows": rows_list
    }

    payload_bytes = json.dumps(payload).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AgenticDermaGuard-Level2Submit/2.0"
    }
    if team_token:
        headers["Authorization"] = f"Bearer {team_token}"
        headers["X-API-KEY"] = team_token

    req = urllib.request.Request(api_url, data=payload_bytes, headers=headers, method="POST")

    print(f"\n[Submitting] Uploading Level 2 submission to {api_url} ...")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            code = response.getcode()
            body = response.read().decode('utf-8')
            print(f"\n[SUCCESS] Server returned HTTP {code}!")
            print(f"[Server Response]: {body}")

            receipt = {
                "level": 2,
                "status_code": code,
                "api_url": api_url,
                "team_name": team_name,
                "response": body,
                "total_rows": len(rows_list)
            }
            with open("outputs/submission_receipt_level2.json", "w") as f:
                json.dump(receipt, f, indent=2)
            print("[Receipt] Saved to outputs/submission_receipt_level2.json")
            return True
    except urllib.error.HTTPError as e:
        print(f"\n[HTTP Error] Server returned HTTP {e.code}: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"\n[Network Error]: {str(e)}")
        return False

def submit_level1_results(api_url: str = None, team_name: str = None, team_token: str = None):
    summary_path = "outputs/validation_results.json"
    if not os.path.exists(summary_path):
        print(f"[Error] Results summary file '{summary_path}' not found! Run evaluate.py first.")
        return False

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    if not api_url:
        api_url = os.environ.get("DERMAGUARD_API_URL") or input("Enter Submission API URL: ").strip()
    if not team_name:
        team_name = os.environ.get("DERMAGUARD_TEAM_NAME", "Team-DermaGuard")
    if not team_token:
        team_token = os.environ.get("DERMAGUARD_TEAM_TOKEN", "")

    payload = {
        "team_name": team_name,
        "track": "Track 2 | Healthcare",
        "level": 2,
        "project": "Agentic DermaGuard (Six-Agent System)",
        "timestamp": summary_data.get("evaluation_timestamp"),
        "total_validation_images": summary_data.get("total_eval_samples"),
        "classification_metrics": summary_data.get("classification_metrics", {}),
        "live_demo_reporting_counts": summary_data.get("live_demo_reporting_counts", {}),
        "six_agent_summary": summary_data.get("six_agent_summary", {})
    }

    payload_bytes = json.dumps(payload).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AgenticDermaGuard-SubmitClient/2.0"
    }
    if team_token:
        headers["Authorization"] = f"Bearer {team_token}"
        headers["X-API-KEY"] = team_token

    req = urllib.request.Request(api_url, data=payload_bytes, headers=headers, method="POST")
    print(f"\n[Submitting] Sending payload to {api_url} ...")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            code = response.getcode()
            body = response.read().decode('utf-8')
            print(f"[Success] HTTP {code}: {body}")
            return True
    except Exception as e:
        print(f"[Error]: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic DermaGuard Summit API Submission Client")
    parser.add_argument("--csv", type=str, default=None, help="Path to Level 2 submission CSV file to upload")
    parser.add_argument("--api-url", type=str, help="Submission API Endpoint")
    parser.add_argument("--team-name", type=str, help="Your Team Name")
    parser.add_argument("--token", type=str, help="Submission API Token / Key")
    args = parser.parse_args()

    if args.csv:
        submit_csv_to_api(csv_path=args.csv, api_url=args.api_url, team_name=args.team_name, team_token=args.token)
    else:
        submit_level1_results(api_url=args.api_url, team_name=args.team_name, team_token=args.token)
