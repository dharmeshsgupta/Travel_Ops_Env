"""
app.py – FastAPI server exposing TravelOpsEnv for Hugging Face Spaces.
Provides REST endpoints for environment reset, step, and grading.
"""
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

from environment import TravelOpsEnv
from models import Action, Observation

app = FastAPI(
    title="TravelOpsEnv API",
    description="Enterprise-grade RL environment for Level 2 travel support agent evaluation.",
    version="1.0.0",
)

# ── Global environment instance ──────────────────────────────────────────────
env: Optional[TravelOpsEnv] = None


# ── Request / Response Schemas ───────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_level: str = Field(default="hard", description="Task difficulty: 'normal' or 'hard'")


class StepRequest(BaseModel):
    action_type: Literal[
        "SEARCH_BOOKINGS", "FETCH_FLIGHT_STATUS", "PROCESS_REFUND",
        "SEARCH_POLICY_DOCS", "MODIFY_BOOKING", "SEND_REPLY", "END_EPISODE",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class GradeResponse(BaseModel):
    score: float
    task_level: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "environment": "TravelOpsEnv",
        "version": "1.0.0",
        "description": "Enterprise-grade RL environment for travel support agent evaluation.",
        "endpoints": ["/reset", "/step", "/grade", "/docs"],
    }


@app.post("/")
def health_check_post():
    """Some validators POST to `/` without a JSON body; mirror GET so probes do not 422."""
    return health_check()


@app.post("/reset", response_model=dict)
def reset_env(req: ResetRequest = None):
    global env
    current_task = req.task_level if req else "hard"
    env = TravelOpsEnv(task_level=current_task)
    obs = env.reset()
    
    return {
        "observation": obs.model_dump(),
        "message": f"Environment reset with task_level={current_task}"
    }


@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    action = Action(action_type=req.action_type, payload=req.payload)
    obs, reward, done, info = env.step(action)

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.post("/grade", response_model=GradeResponse)
def grade_env():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if env.task_level == "hard":
        score = env.grade_hard_task()
    else:
        score = env.grade()

    return GradeResponse(score=score, task_level=env.task_level)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

