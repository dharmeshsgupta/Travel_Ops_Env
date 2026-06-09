import os
import json
import sys

# Line-buffer stdout so judges see [START]/[STEP]/[END] immediately (not stuck in buffer).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# Scaler / hackathon runs LLM calls through their LiteLLM proxy. They observe traffic on
# API_KEY. Using only HF_TOKEN or a hardcoded OpenAI URL bypasses the proxy and fails
# "LLM Criteria Check" (no API calls on the provided key).
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")


def main():
    step_n = 0
    rewards_list = []
    success = False
    env = None

    print(
        f"[START] task=travel-support env=TravelOpsEnv model={MODEL_NAME}",
        flush=True,
    )

    # Import here so [START] is always printed first; keep openai in requirements.txt for Phase 2.
    try:
        from openai import OpenAI
        from openenv import SyncEnvClient
    except ModuleNotFoundError as e:
        print(f"Error: {e}", flush=True)
        print("[END] success=false steps=0 rewards=0.00", flush=True)
        raise SystemExit(1) from e

    api_base_url = os.getenv("API_BASE_URL")
    # Prefer hackathon-injected key so requests hit the LiteLLM proxy they monitor.
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        print("[END] success=false steps=0 rewards=0.00", flush=True)
        raise SystemExit(
            "Set API_KEY (hackathon) or HF_TOKEN (local/HF) for the LLM client."
        )
    if not api_base_url:
        # Local dev fallback only — evaluation sets API_BASE_URL to the proxy URL.
        api_base_url = "https://api.openai.com/v1"

    try:
        client = OpenAI(base_url=api_base_url, api_key=api_key)
        env = SyncEnvClient("https://dharmeshsgupta-travel-ops-env.hf.space")

        obs, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated and step_n < 10:
            step_n += 1
            error_msg = "null"

            prompt = (
                f"Given this state: {json.dumps(obs)}, what should I do next? "
                'Just output a valid JSON action string like {"action_type": "SEARCH_BOOKINGS", '
                '"payload": {"user_id": 103}}'
            )

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                action_str = response.choices[0].message.content
                action_dict = json.loads(action_str)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                action_dict = {"action_type": "END_EPISODE", "payload": {}}
                action_str = "parsing_error"
                done = True

            try:
                obs, reward, done, truncated, info = env.step(action_dict)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                reward = 0.0
                done = True
                truncated = True

            rewards_list.append(reward)

            action_clean = action_str.replace("\n", "").replace("\r", "")
            done_str = "true" if (done or truncated) else "false"

            print(
                f"[STEP] step={step_n} action={action_clean} reward={reward:.2f} "
                f"done={done_str} error={error_msg}",
                flush=True,
            )

            if reward >= 1.0:
                success = True

    except Exception as e:
        print(f"Error: {e}", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        success_str = "true" if success else "false"
        rewards_str = (
            ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
        )
        print(
            f"[END] success={success_str} steps={step_n} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
