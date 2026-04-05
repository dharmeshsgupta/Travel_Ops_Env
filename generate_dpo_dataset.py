"""
DPO Dataset Generator for TravelOpsEnv
───────────────────────────────────────
Runs the baseline agent logic for N episodes on the Hard Task,
captures full interaction trajectories, and exports a Hugging Face
DPO-ready JSONL file (travelops_dpo.jsonl).

Usage:
    python generate_dpo_dataset.py                  # 5 simulated episodes
    python generate_dpo_dataset.py --live           # 5 live LLM episodes (needs HF_TOKEN)
    python generate_dpo_dataset.py --episodes 20    # custom count
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from environment import TravelOpsEnv
from models import Action


# ── Simulated Agent Strategies ────────────────────────────────────────────────
def _run_good_agent(env, charlie_data):
    """Expert trajectory: paginate → RAG → refund → reply → end."""
    bkg_id = charlie_data["booking_id"]
    flt_id = charlie_data["flight_id"]
    messages = []

    plan = [
        {"action_type": "SEARCH_BOOKINGS",    "payload": {"user_id": 103}},
        {"action_type": "SEARCH_BOOKINGS",    "payload": {"user_id": 103, "cursor": 2}},
        {"action_type": "SEARCH_POLICY_DOCS", "payload": {"query": "refund"}},
        {"action_type": "FETCH_FLIGHT_STATUS","payload": {"flight_id": flt_id}},
        {"action_type": "PROCESS_REFUND",     "payload": {"booking_id": bkg_id, "amount": 2200}},
        {"action_type": "SEND_REPLY",         "payload": {"message": f"Hi Charlie, your refund of $2200 for booking {bkg_id} has been processed. The delay on flight {flt_id} qualifies you for a full refund under our policy."}},
        {"action_type": "END_EPISODE",        "payload": {}},
    ]

    done = False
    plan_idx = 0
    total_reward = 0.0

    while not done and plan_idx < len(plan):
        action_json = plan[plan_idx]
        ai_text = json.dumps(action_json)
        messages.append({"role": "assistant", "content": ai_text})

        action = Action(**action_json)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Retry on 503 gateway errors
        if "503" in obs.system_feedback:
            messages.append({"role": "user", "content": obs.model_dump_json()})
            continue

        if not done:
            messages.append({"role": "user", "content": obs.model_dump_json()})
        plan_idx += 1

    return messages, total_reward


def _run_bad_agent(env, charlie_data):
    """Naive trajectory: skip pagination & RAG, jump straight to refund."""
    bkg_id = charlie_data["booking_id"]
    messages = []

    plan = [
        {"action_type": "PROCESS_REFUND", "payload": {"booking_id": bkg_id, "amount": 2200}},
        {"action_type": "END_EPISODE",    "payload": {}},
    ]

    done = False
    plan_idx = 0
    total_reward = 0.0

    while not done and plan_idx < len(plan):
        action_json = plan[plan_idx]
        ai_text = json.dumps(action_json)
        messages.append({"role": "assistant", "content": ai_text})

        action = Action(**action_json)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if not done:
            messages.append({"role": "user", "content": obs.model_dump_json()})
        plan_idx += 1

    return messages, total_reward


# ── Live Agent (uses HF Router) ──────────────────────────────────────────────
def _run_live_agent(env):
    """Run a real LLM agent via the HF Router."""
    from openai import OpenAI

    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    api_base = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_id = os.getenv("MODEL_ID") or "Qwen/Qwen2.5-72B-Instruct"

    client = OpenAI(api_key=api_key, base_url=api_base)

    system_prompt = (
        "You are an AI customer support agent. Respond ONLY with valid JSON:\n"
        '{"action_type": "...", "payload": {...}}\n'
        "Available actions: SEARCH_BOOKINGS, FETCH_FLIGHT_STATUS, PROCESS_REFUND, "
        "SEARCH_POLICY_DOCS, SEND_REPLY, END_EPISODE."
    )

    obs = env.reset()
    done = False
    messages_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": obs.model_dump_json()},
    ]
    trajectory = []
    total_reward = 0.0

    while not done:
        resp = client.chat.completions.create(
            model=model_id,
            response_format={"type": "json_object"},
            messages=messages_llm,
        )
        ai_text = resp.choices[0].message.content
        trajectory.append({"role": "assistant", "content": ai_text})
        messages_llm.append({"role": "assistant", "content": ai_text})

        try:
            action = Action(**json.loads(ai_text))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if not done:
                obs_json = obs.model_dump_json()
                trajectory.append({"role": "user", "content": obs_json})
                messages_llm.append({"role": "user", "content": obs_json})
        except Exception:
            done = True

    return trajectory, total_reward


# ── Main Generator ────────────────────────────────────────────────────────────
def generate(num_episodes: int = 5, live: bool = False):
    results = []  # (score, prompt_msgs, trajectory_msgs)

    print(f"Collecting {num_episodes} episodes for DPO dataset ({'live LLM' if live else 'simulated'})...\n")

    for ep in range(num_episodes):
        env = TravelOpsEnv(task_level="hard")

        if live:
            obs = env.reset()  # already called inside _run_live_agent, but we need initial obs for prompt
            env2 = TravelOpsEnv(task_level="hard")
            obs2 = env2.reset()
            prompt_msgs = [
                {"role": "system", "content": "You are an AI customer support agent."},
                {"role": "user",   "content": obs2.model_dump_json()},
            ]
            trajectory, total_reward = _run_live_agent(env2)
            score = env2.grade_hard_task()
        else:
            obs = env.reset()
            charlie_data = env.test_data["charlie"]

            prompt_msgs = [
                {"role": "system", "content": "You are an AI customer support agent."},
                {"role": "user",   "content": obs.model_dump_json()},
            ]

            # Alternate between good and bad agent strategies
            if ep % 2 == 0:
                trajectory, total_reward = _run_good_agent(env, charlie_data)
            else:
                trajectory, total_reward = _run_bad_agent(env, charlie_data)

            score = env.grade_hard_task()

        label = "chosen" if score > 0.8 else ("rejected" if score < 0.5 else "neutral")
        print(f"  Episode {ep+1:>2}/{num_episodes}  |  Score: {score:>5.2f}  |  Reward: {total_reward:>+6.2f}  |  Label: {label}")

        results.append({
            "score": score,
            "label": label,
            "prompt": prompt_msgs,
            "trajectory": trajectory,
        })

    # ── Build DPO pairs ───────────────────────────────────────────────
    chosen   = [r for r in results if r["label"] == "chosen"]
    rejected = [r for r in results if r["label"] == "rejected"]

    print(f"\n  Chosen: {len(chosen)}  |  Rejected: {len(rejected)}")

    pairs = []
    for c in chosen:
        for rej in rejected:
            pairs.append({
                "prompt":   c["prompt"],
                "chosen":   c["trajectory"],
                "rejected": rej["trajectory"],
            })

    outfile = "travelops_dpo.jsonl"
    if pairs:
        with open(outfile, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n✓ Wrote {len(pairs)} DPO pairs to {outfile}")
    else:
        print("\n⚠ Not enough chosen/rejected trajectories. Increase --episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO dataset from TravelOpsEnv")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--live", action="store_true", help="Use live LLM via HF Router")
    args = parser.parse_args()
    generate(num_episodes=args.episodes, live=args.live)
