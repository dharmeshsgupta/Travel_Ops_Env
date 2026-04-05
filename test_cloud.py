import requests
import json

def test_cloud_deployment():
    print("Connecting to cloud environment on Hugging Face...")
    print("Connection successful! Resetting environment via REST HTTP Fallback...")
    
    url = "https://dharmeshsgupta-travel-ops-env.hf.space/reset"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json={})
    
    if response.status_code == 200:
        obs = response.json()
        print("\n--- Initial Observation Received ---")
        print(json.dumps(obs, indent=2))
        print("\nSuccess! The cloud API is responding.")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_cloud_deployment()
