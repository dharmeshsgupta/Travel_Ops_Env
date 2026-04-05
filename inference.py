import os
import json
from openai import OpenAI
from openenv import SyncEnvClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN must be set")
        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SyncEnvClient("https://dharmeshsgupta-travel-ops-env.hf.space")
    
    print("START")
    
    try:
        obs, info = env.reset()

        for _ in range(5):
            print("STEP")

            prompt = f"Given this state: {json.dumps(obs)}, what should I do next? Just output a valid JSON action string like {{'action_type': 'SEARCH_BOOKINGS', 'parameters': {{'user_id': 103}}}}"

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            print(response.choices[0].message.content)

            obs, reward, done, truncated, info = env.step({"action_type": "END_EPISODE", "parameters": {}})

            if done or truncated:
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("END")

if __name__ == '__main__':
    main()
