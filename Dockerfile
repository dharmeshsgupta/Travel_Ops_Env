FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends sqlite3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY db_setup.py       /app/db_setup.py
COPY models.py         /app/models.py
COPY environment.py    /app/environment.py
COPY openenv.yaml      /app/openenv.yaml
COPY generate_dpo_dataset.py /app/generate_dpo_dataset.py
COPY train_ppo.py      /app/train_ppo.py

# Baseline agent
COPY travel_ops_env/   /app/travel_ops_env/

# ── Health-check & entrypoint ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "from environment import TravelOpsEnv; e=TravelOpsEnv(); e.reset(); print('OK')"

# Default: validate the environment then keep the container alive for the judges
CMD ["python", "-c", "\
from environment import TravelOpsEnv; \
env = TravelOpsEnv(task_level='hard'); \
obs = env.reset(); \
print('TravelOpsEnv ready on Hugging Face Spaces'); \
print(f'Initial observation: {obs.system_feedback}'); \
"]
