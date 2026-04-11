import os
import json
from openai import OpenAI
from openenv import SyncEnvClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

def main():        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Use the official SDK as requested by the Hackathon rules
    env = SyncEnvClient("https://dharmeshsgupta-travel-ops-env.hf.space")
    
    # ---------------------------------------------------------
    # 1. EMIT THE EXACT [START] TAG FORMATTING
    # ---------------------------------------------------------
    print(f"[START] task=travel-support env=TravelOpsEnv model={MODEL_NAME}")
    
    try:
        # Reset Env via official SDK
        obs, info = env.reset()
        
        rewards_list = []
        step_n = 0
        done = False
        truncated = False
        success = False

        # Run for a maximum of 10 steps to prevent infinite loop
        while not done and not truncated and step_n < 10:
            step_n += 1
            error_msg = "null"

            prompt = f"Given this state: {json.dumps(obs)}, what should I do next? Just output a valid JSON action string like {{\"action_type\": \"SEARCH_BOOKINGS\", \"payload\": {{\"user_id\": 103}}}}"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                action_str = response.choices[0].message.content
                action_dict = json.loads(action_str)
            except Exception as e:
                # Fallback if agent hallucinates or crashes
                error_msg = str(e).replace("\n", " ")
                action_dict = {"action_type": "END_EPISODE", "payload": {}}
                action_str = "parsing_error"
                done = True

            # Step Env via official SDK
            try:
                obs, reward, done, truncated, info = env.step(action_dict)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                reward = 0.0
                done = True
                truncated = True
                
            rewards_list.append(reward)
            
            # Format outputs properly for stdout grading
            action_clean = action_str.replace("\n", "").replace("\r", "")
            done_str = "true" if (done or truncated) else "false"
            
            # ---------------------------------------------------------
            # 2. EMIT THE EXACT [STEP] TAG FORMATTING
            # ---------------------------------------------------------
            print(f"[STEP] step={step_n} action={action_clean} reward={reward:.2f} done={done_str} error={error_msg}")

            if reward >= 1.0:
                success = True

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            env.close()
        except:
            pass

        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
        
        # ---------------------------------------------------------
        # 3. EMIT THE EXACT [END] TAG FORMATTING
        # ---------------------------------------------------------
        print(f"[END] success={success_str} steps={step_n} rewards={rewards_str}")

if __name__ == '__main__':
    main()
