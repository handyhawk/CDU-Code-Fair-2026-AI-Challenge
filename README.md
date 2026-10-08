# HumanFirst AI

An AI-assisted triage system for a government inbox. It reads incoming
messages, decides how urgent each one is, sorts it into a category and a
team, and sends the risky or uncertain ones to a person to check.
**The AI assists; a human always decides.**

Built for the CDU IT Code Fair 2026 AI Challenge, scenario 5: getting
urgent messages to a human fast.

---

## 1. How the system works

```text
Citizen message / CSV
        │
        ▼
Flask backend
        │
        ▼
OpenAI classifier
        │
        ├── if unavailable ──► Local ML fallback
        │
        ▼
Independent Safety Engine
        │
        ▼
SQLite case storage
        │
        ▼
Staff review queue
        │
        ▼
Human confirm / override
```

### What it produces for each message

| Field        | Values                                                                                   |
|--------------|------------------------------------------------------------------------------------------|
| **Urgency**  | `Critical`, `High`, `Normal`                                                              |
| **Category** | Health & Safety, Housing & Utilities, Financial Support, Licensing & Services, General Enquiries |
| **Route**    | The team that should handle it. Usually the same as the category, but it can differ (a housing issue with a medical angle goes to Health & Safety) |
| **Engine**   | `OpenAI` (primary) or `Local ML` (fallback)                                              |
| **Safety**   | Independent Safety Engine check, including any escalation reason                            |

### Hybrid classification and independent safety layer

1. **OpenAI is the primary classifier.** Each message is sent to
   `gpt-5-mini`, which returns a structured result (urgency, category,
   route and a one-sentence explanation). It does not return a calibrated
   probability, so the app shows confidence as **N/A** rather than
   inventing a number. Critical and High cases are escalated to a human.
2. **A local model is the fallback.** If OpenAI cannot be reached because of
   internet, API-key, quota or service issues, the backend uses a local
   TF-IDF + logistic regression model trained on
   `backend/data/HumanFirst_training_messages_v3.csv`. The v3 corpus contains
   600 synthetic labelled messages. Fallback results are marked
   **Provisional** and are **always sent to human review**.
3. **The Safety Engine is independent of both classifiers.** After either
   OpenAI or Local ML produces an urgency result, the Safety Engine checks the
   original message for known high-risk circumstances and combinations. It can
   raise urgency but never lower it. Safety findings and reasons are stored
   with the case for staff review.

### Tone is not evidence

Words like "URGENT", capital letters and "!!!" do not raise the urgency.
The app flags them as attention markers for the reviewer but never scores
them. Urgency comes from the circumstances described: medication
dependency, children, loss of essential services, threats, deadlines and
so on. The test set includes `URGENT!!! I forgot my password 😂`, which is
correctly `Normal`.

### Human review

Every case lands in the **Review queue**, most urgent first. A reviewer
can confirm the AI's call or override it with a reason. Both decisions are
saved with the case, so the original AI result and the final human
decision are always visible side by side. Cases are stored in
`backend/data/cases.db` and survive restarts.

### Two portals, one app

The frontend has two views. Switch with the button in the top-right corner,
or open them directly with `?view=citizen` (the default) or `?view=staff`.

**Citizen portal** (<http://localhost:8501/?view=citizen>): a member of the
public writes a message and gets a confirmation with a reference number such as
`HF-000123`. They never see the AI's assessment.

**Staff portal** (<http://localhost:8501/?view=staff>):

| Page             | Purpose                                                                                   |
|------------------|-------------------------------------------------------------------------------------------|
| **Review queue** | Cases awaiting a human, most urgent first. Search by reference (`HF-000123` or just `123`), case details, AI and Safety Engine result, confirm or override |
| **Batch upload** | Upload a CSV of messages and watch live progress as each one is classified                 |
| **Dashboard**    | Counts by urgency, category, review status and analysis engine                             |

The same reference number appears in both portals. The staff **Settings** button
holds the dark mode toggle, the backend address, a CSV export and a demo reset.

---

## 2. How to set up the application

### Requirements

- Python 3.10 or newer
- An OpenAI API key is recommended for the primary classifier
- Internet access is required for the OpenAI path; without it, the application
  continues using the Local ML fallback

> **Windows tip:** keep the project in a plain folder such as
> `C:\Dev\...`, not inside OneDrive. OneDrive syncing can corrupt files
> while Python is running.

### Step 1: Get the code

```bash
git clone https://github.com/handyhawk/cdu-code-fair-2026-ai-challenge.git
cd cdu-code-fair-2026-ai-challenge
```

### Step 2: Configure the OpenAI API key (recommended)

1. Sign in at <https://platform.openai.com>.
2. Configure API billing if required for the account. The API is billed
   separately from a ChatGPT subscription.
3. Open <https://platform.openai.com/api-keys> and choose
   **Create new secret key**. Name it (for example `humanfirst-demo`).
4. Copy the key when it is created and store it securely.

If no API key is configured, HumanFirst AI still starts and uses the Local ML
fallback. Those results are marked **Provisional** and require human review.

Treat the key like a password:

- Never commit it, paste it into chat, or include it in screenshots or
  logs. If it leaks, revoke it on the same page and create a new one.
- Each team member should use their own key.

### Step 3: Create the `.env` file for OpenAI

If the OpenAI primary classifier is being used, place the key in a file named
`.env` **inside the `backend` folder** (next to `app.py`). The file should be
excluded from Git so credentials are never committed.

```
HumanFirst AI/
├── README.md
├── backend/
│   ├── app.py
│   ├── .env              ← create this file here
│   ├── requirements.txt
│   └── ...
└── frontend/
    └── app.py
```

The file contains one line, with no quotes and no spaces around `=`:

```
OPENAI_API_KEY=sk-your-key-here
```

Create it from a terminal opened in the project root:

```powershell
# Windows PowerShell
Set-Content -Path backend\.env -Value "OPENAI_API_KEY=sk-your-key-here"
```
```bash
# macOS / Linux
echo "OPENAI_API_KEY=sk-your-key-here" > backend/.env
```

Or create a new file called `.env` in your editor. Make sure it is not
saved as `.env.txt`.

### Step 4: Install and start the backend

```bash
cd backend
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On start-up you should see the local fallback model train, then Flask
listening on `http://localhost:5000`. Leave this terminal running.

Check it: open <http://localhost:5000/health> in a browser. You should
see `"status": "ok"`.

### Step 5: Install and start the frontend

Open a **second** terminal in the project root:

```bash
cd frontend
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Both terminals must stay
running. It opens on the citizen portal; choose **Staff portal** in the top
right, which should read *Service online*.

### Try it

Send this from the citizen portal:

> My electricity was disconnected this morning and my insulin has to stay
> refrigerated. I don't know what to do.

Expected: a confirmation with a reference number. In the staff portal the same
reference is at the top of the **Review queue**: `Critical`, classified by
`OpenAI`, awaiting human review.

### Troubleshooting

| Symptom                                                    | Cause and fix                                                                                   |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Every case shows **Local ML** and **Provisional**          | OpenAI is unavailable or no API key is configured. Check the backend terminal for `[OpenAI unavailable]`. |
| Frontend says the backend is not reachable                 | Backend is not running, or the address in **Settings → Backend address** is wrong.                |
| Changes to `.env` have no effect                           | Restart the backend. The key is read once at start-up.                                           |
| Theme or font changes do not appear                        | Restart `streamlit run`. `.streamlit/config.toml` is read only at start-up.                      |

---

## 3. How to configure the main script

The main backend script is `backend/app.py`. These are the settings you
are most likely to change.

### Environment (`backend/.env`)

| Variable         | Required | Meaning                          |
|------------------|----------|----------------------------------|
| `OPENAI_API_KEY` | Optional | Enables the OpenAI primary classifier; without it the Local ML fallback is used |

### Backend settings

| What                          | Where                                                    | Notes                                                                 |
|-------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------|
| OpenAI model                  | `model="gpt-5-mini"` in `classify_message()`, `openai_analyzer.py` | Swap for another model that supports structured output                |
| Classification rules          | `SYSTEM_PROMPT` in `openai_analyzer.py`                  | Defines Critical / High / Normal and the "tone is not evidence" rule  |
| Categories and teams          | `Category` enum in `openai_analyzer.py`                  | Keep in step with `ESCALATION_MAP` and the training CSV               |
| Fallback training data        | `DEFAULT_TRAINING_CSV` in `app.py`                       | A labelled CSV the Local ML model trains on at start-up               |
| Server port and host          | `app.run(...)` at the bottom of `app.py`                 | Default `127.0.0.1:5000` with `debug=False` |
| Fallback escalation threshold | `ESCALATION_CONFIDENCE_THRESHOLD` in `urgency_analyzer.py` | Default `0.45`. Lower sends fewer low-confidence cases to a human     |
| Risk keywords                 | `SUBSTANTIVE_RISK_KEYWORDS` in `urgency_analyzer.py`     | Real circumstances that may raise the fallback model's urgency        |
| Attention markers             | `ATTENTION_MARKER_WORDS` in `urgency_analyzer.py`        | Shown to reviewers but never scored                                   |
| Database location             | `DEFAULT_DB_PATH` in `database.py`                       | Default `backend/data/cases.db`                                       |

### Retrain the fallback model without restarting

```bash
curl -X POST http://localhost:5000/train -F "file=@path/to/new_cases.csv"
```

### Frontend settings

| What                 | Where                                          | Notes                                              |
|----------------------|------------------------------------------------|----------------------------------------------------|
| Backend address      | **Settings → Backend address**, or `DEFAULT_BACKEND_URL` in `frontend/app.py` | Default `http://localhost:5000`                    |
| Theme, font, colours | `frontend/.streamlit/config.toml` and the `LIGHT_VARS` / `DARK_VARS` blocks in `frontend/app.py` | Restart Streamlit after editing the TOML file      |
| Frontend port        | `streamlit run app.py --server.port 8600`      | Default `8501`                                     |

### Batch CSV format

The CSV needs a column holding the message text. The loader accepts the
common names (for example `message`, `text`, `message_text`, `content`); see
`COLUMN_ALIASES` in `backend/data_loader.py` for the full list. Each row
is classified one at a time, so large files take as long as the OpenAI
calls do.

### API reference

| Method and path             | Purpose                                         |
|-----------------------------|-------------------------------------------------|
| `GET /health`               | Backend status and whether the fallback is trained |
| `POST /triage`              | Classify one message: `{"message": "..."}`      |
| `POST /analyze`             | Classify a whole CSV (`file` field)             |
| `POST /train`               | Retrain the fallback model from a CSV           |
| `GET /cases`                | List stored cases (filters: `urgency`, `category`, `escalate_to_human`, `limit`) |
| `GET /cases/summary`        | Counts for the dashboard                        |
| `POST /cases/<id>/review`   | Save a human decision: `human_urgency`, `human_notes` |
| `DELETE /cases`             | Clear all stored cases (demo reset)             |

---

## Project layout

```
backend/    Flask API, OpenAI classifier, local ML fallback, SQLite storage
frontend/   Streamlit interface
```

## Team

Andy (project lead, evaluation) · Minh (AI and backend) · Kien (application and UI)
