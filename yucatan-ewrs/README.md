# Yucatán EWRS – Hurricane Early Warning & Response System

A Streamlit MVP built for a Digital Economy presentation. It shows how an AI layer (hyperlocal ML risk scoring, multilingual LLM alerts and human-in-the-loop triage) can sit on top of the simple protocols Yucatán already uses: Cell Broadcast, community radio, printed maps and drainage maintenance.

> **Works 100% offline.** If no API key is set, the LLM call fails or times out, Offline / Demo Mode is on, or a telecom outage is simulated, the app switches to the pre-approved templates in `data/fallback_alerts.json`.

---

## Quick start (Linux / macOS)

```bash
cd yucatan-ewrs

# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) configure an LLM provider
cp .env.example .env        # then edit .env and add GEMINI_API_KEY or OPENAI_API_KEY

# 4. (Optional) pre-train the model; the app also trains it on first launch
python -m models.risk_model

# 5. Verify everything before the presentation
python scripts/verify.py

# 6. Run the app
streamlit run app.py
```

Then open http://localhost:8501.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts\verify.py
streamlit run app.py
```

If you'd rather not activate the venv, call its interpreter directly: `.venv/bin/streamlit run app.py`.

---

## Run with Docker (recommended for the live demo)

The model is trained while the image builds, so the container starts in seconds and needs no internet once built.

```bash
cd yucatan-ewrs

# (Optional) add API keys. Without a .env the app runs in Offline / Demo Mode.
cp .env.example .env

# Build the image and start the container in the background
docker compose up --build -d

# Check status (wait for "healthy") and follow logs
docker compose ps
docker compose logs -f

# Stop and remove the container
docker compose down
```

Then open http://localhost:8501.

- To run in the foreground instead, use `docker compose up --build` and stop it with `Ctrl+C`.
- `.env` is optional (`required: false`). If you add or change it, apply it with `docker compose up -d --force-recreate`.
- The container restarts automatically (`unless-stopped`), including after a Docker or machine restart, until you run `docker compose down`.
- **Before presenting offline:** run `docker compose build` while you still have internet, because the build downloads the base image and Python packages.

Plain Docker, without Compose:

```bash
docker build -t yucatan-ewrs .
docker run --rm -p 8501:8501 --env-file .env yucatan-ewrs   # omit --env-file if there is no .env
```

---

## LLM configuration (`.env`)

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` or `openai` | `gemini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Google AI Studio key / model id | `gemini-2.5-flash` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI key / model id | `gpt-4o-mini` |

With no key, Offline / Demo Mode is on by default. Each generated alert has a source badge, either **Live LLM** or **Offline Fallback**, plus the reason for any fallback.

---

## Project structure

```
yucatan-ewrs/
├── app.py                      # Streamlit UI: sidebar plus 4 tabs
├── requirements.txt
├── Dockerfile                  # python:3.11-slim, model baked in at build time, healthcheck
├── docker-compose.yml          # port 8501, optional .env, restart unless-stopped
├── .dockerignore
├── .env.example
├── .streamlit/config.toml      # neutral theme, telemetry off (offline-friendly)
├── data/
│   ├── municipalities.json     # 8 municipalities, approximate baseline indicators
│   ├── fallback_alerts.json    # pre-approved ES / Maya / EN templates (4 risk levels × 3 channels)
│   └── sample_incidents.json   # 10 synthetic post-impact incidents and the brigade roster
├── models/
│   ├── risk_model.py           # synthetic data, RandomForest, joblib, predict_municipality_risks()
│   └── artifacts/              # risk_model.joblib (generated)
├── services/
│   ├── alert_generator.py      # Gemini / OpenAI with automatic template fallback
│   └── triage_engine.py        # priority scoring, brigade allocation, HITL decisions
└── scripts/
    └── verify.py               # pre-demo health check
```

## The four tabs

1. **Hyperlocal Risk Map.** KPI cards, an OpenStreetMap risk map with the storm track (it falls back to an offline vector map automatically), a risk ranking, and the top 3 ML risk drivers for each municipality.
2. **Multilingual Alert Studio.** Cell Broadcast SMS (160 characters or fewer), a WhatsApp message and a 30-second radio script, each in Spanish, Yucatec Maya and English.
3. **Post-Impact Triage.** A priority queue with AI-suggested brigades. A PROCIVY operator approves or overrides each one, and every decision goes into an audit log.
4. **Critical Check.** Compares the baseline protocols with the AI layer. Turn on *Simulate Power & Telecom Outage* to see which parts degrade.

## Suggested demo script (≈5 min)

1. Start with a Cat 4 storm, 300 mm of rain, 3 m surge, landfall on the North Coast. Walk through the KPIs and the map.
2. Change the landfall zone to East Coast. Risk shifts to Río Lagartos and El Cuyo. Open the drivers table.
3. Go to Alert Studio, pick El Cuyo and click Generate. Show the three languages and point out the Maya validation disclaimer.
4. Go to Triage. Approve one suggestion, override another with a reason, then open the audit log.
5. Turn on **Simulate Power & Telecom Outage**. In Critical Check, the AI layer degrades while radio, printed maps and drainage keep working.

## Disclaimers

- Baseline indicators are rounded approximations for the prototype. Real-world sources are INEGI (Censo 2020), CONAGUA / SMN and PROCIVY.
- The risk model is trained on synthetic but physically plausible scenarios. It is not an operational forecast.
- Yucatec Maya outputs must be validated by native speakers and INDEMAYA before any real deployment.
- All incidents are fictional.
