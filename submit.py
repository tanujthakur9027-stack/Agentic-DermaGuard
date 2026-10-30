import os
import sys
import json
import argparse
import urllib.request
import urllib.error

def submit_level1_results(api_url: str = None, team_name: str = None, team_token: str = None):
    summary_path = "outputs/validation_results.json"
    if not os.path.exists(summary_path):
        print(f"[Error] Results summary file '{summary_path}' not found! Run evaluate.py first.")
        return False
        
    with open(summary_path, "r") as f:
        summary_data = json.load(f)
        
    if not api_url:
        api_url = os.environ.get("DERMAGUARD_API_URL")
    if not team_name:
        team_name = os.environ.get("DERMAGUARD_TEAM_NAME", "Team-DermaGuard")
    if not team_token:
        team_token = os.environ.get("DERMAGUARD_TEAM_TOKEN", "")

    print("=" * 60)
    print("  JAYPEE AGENTIC AI INTERNATIONAL SUMMIT 2026")
    print("  Track 2: Healthcare - Level 1 Submission Client")
    print("=" * 60)
    
    if not api_url:
        print("\n[Input Required] The submission API URL was shared in your summit WhatsApp group.")
        api_url = input("Enter the Submission API URL: ").strip()
        
    if not team_name or team_name == "Team-DermaGuard":
        user_team = input(f"Enter your Team Name [{team_name}]: ").strip()
        if user_team:
            team_name = user_team
            
    if not team_token:
        team_token = input("Enter your Team Token / API Key (if provided, else leave blank): ").strip()

    # Construct submission payload
    payload = {
        "team_name": team_name,
        "track": "Track 2 | Healthcare",
        "level": 1,
        "project": "Agentic DermaGuard",
        "timestamp": summary_data.get("evaluation_timestamp"),
        "total_validation_images": summary_data.get("total_eval_samples"),
        "baseline_metrics": {
            "overall_accuracy": summary_data["baseline_classification_metrics"]["overall_accuracy"],
            "balanced_accuracy": summary_data["baseline_classification_metrics"]["balanced_accuracy"],
            "macro_f1": summary_data["baseline_classification_metrics"]["macro_f1_score"],
            "weighted_f1": summary_data["baseline_classification_metrics"]["weighted_f1_score"]
        },
        "image_quality_summary": summary_data.get("image_quality_agent_outcomes"),
        "triage_summary": summary_data.get("multi_agent_triage_decisions"),
        "model_architecture": "EfficientNet-B0 + Laplacian/Entropy/HSV/Canny Multi-Agent"
    }

    print("\n[Submission Payload Preview]:")
    print(json.dumps(payload, indent=2))
    
    confirm = input("\nProceed with API submission? (y/n) [y]: ").strip().lower()
    if confirm not in ('', 'y', 'yes'):
        print("[Aborted] Submission cancelled by user.")
        return False
        
    payload_bytes = json.dumps(payload).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AgenticDermaGuard-SubmitClient/1.0"
    }
    if team_token:
        headers["Authorization"] = f"Bearer {team_token}"
        headers["X-API-KEY"] = team_token
        
    req = urllib.request.Request(api_url, data=payload_bytes, headers=headers, method="POST")
    
    print(f"\n[Submitting] Sending summary payload to {api_url} ...")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_code = response.getcode()
            res_body = response.read().decode('utf-8')
            print(f"\n[Success] Received HTTP {res_code} from submission server!")
            print(f"[Server Response]: {res_body}")
            
            # Save receipt
            receipt = {
                "status_code": res_code,
                "api_url": api_url,
                "team_name": team_name,
                "response": res_body,
                "submitted_payload": payload
            }
            with open("outputs/submission_receipt.json", "w") as f:
                json.dump(receipt, f, indent=2)
            print("[Receipt] Saved to outputs/submission_receipt.json")
            return True
    except urllib.error.HTTPError as e:
        print(f"\n[HTTP Error] Server returned HTTP {e.code}: {e.read().decode('utf-8')}")
        return False
    except urllib.error.URLError as e:
        print(f"\n[Network Error] Failed to connect to server: {e.reason}")
        return False
    except Exception as e:
        print(f"\n[Error] Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic DermaGuard Level 1 API Submission Client")
    parser.add_argument("--api-url", type=str, help="Submission API Endpoint")
    parser.add_argument("--team-name", type=str, help="Your Team Name")
    parser.add_argument("--token", type=str, help="Submission API Token / Key")
    args = parser.parse_args()
    
    submit_level1_results(api_url=args.api_url, team_name=args.team_name, team_token=args.token)
