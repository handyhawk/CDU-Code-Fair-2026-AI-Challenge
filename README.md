# HumanFirst AI

An AI-assisted triage system for a government inbox. It reads incoming
messages, decides how urgent each one is, sorts it into a category and a
team, and sends the risky or uncertain ones to a person to check.
**The AI assists; a human always decides.**

Built for the CDU IT Code Fair 2026 AI Challenge, scenario 5: getting
urgent messages to a human fast.

---

## 1. How the system works

```
                 ┌─────────────────────────────┐
 message ──────► │  Flask backend (port 5000)  │ ──► saved as a "case" (SQLite)
 (typed or CSV)  │                             │              │
                 │  1. OpenAI classifier       │              ▼
                 │  2. Local ML fallback       │     Streamlit frontend (port 8501)
                 └─────────────────────────────┘     triage · review · batch · dashboard
```

### What it produces for each message

| Field        | Values                                                                                   |
|--------------|------------------------------------------------------------------------------------------|
| **Urgency**  | `Critical`, `High`, `Normal`                                                              |
| **Category** | Health & Safety, Housing & Utilities, Financial Support, Licensing & Services, General Enquiries |
| **Route**    | The team that should handle it. Usually the same as the category, but it can differ (a housing issue with a medical angle goes to Health & Safety) |
| **Engine**   | `OpenAI` (primary) or `Local ML` (fallback)                                              |

### Two engines, one safety rule

1. **OpenAI is the primary classifier.** Each message is sent to
   `gpt-5-mini`, which returns a structured result (urgency, category,
   route and a one-sentence explanation). It does not return a calibrated
   probability, so the app shows confidence as **N/A** rather than
   inventing a number. Critical and High cases are escalated to a human.
2. **A local model is the fallback.** If OpenAI cannot be reached (no
   internet, quota used up, bad key, timeout), the backend uses a local
   TF-IDF + logistic regression model trained on
   `backend/data/HumanFirst_AI_30_Test_Messages.csv`. Its result is
   marked **Provisional** and is **always sent to human review**.

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

### Frontend pages

| Page                 | Purpose                                                                      |
|----------------------|------------------------------------------------------------------------------|
| **Triage a message** | Paste one message and see urgency, category, route, engine and explanation    |
| **Review queue**     | Work through cases awaiting a human, filter them, confirm or override         |
| **Batch upload**     | Upload a CSV of messages and watch live progress as each one is classified    |
| **Dashboard**        | Counts by urgency, category and review status                                 |

The **Settings** button on each page holds the dark mode toggle, the backend address, a CSV export and a demo reset.

---

## 2. How to set up the application

### Requirements

- Python 3.10 or newer
- An OpenAI account with an API key (step 2 below)
- Internet access (for OpenAI and for the interface font; the app still
  works offline with the fallback font and the Local ML engine)

> **Windows tip:** keep the project in a plain folder such as
> `C:\Dev\...`, not inside OneDrive. OneDrive syncing can corrupt files
> while Python is running.

### Step 1: Get the code

```bash
git clone https://github.com/handyhawk/cdu-code-fair-2026-ai-challenge.git
cd cdu-code-fair-2026-ai-challenge
```

### Step 2: Get an OpenAI API key

1. Sign in at <https://platform.openai.com>.
2. Add a payment method and a few dollars of credit under **Billing**.
   The API is billed separately from a ChatGPT subscription or just use a
   free API key for testing.
4. Open <https://platform.openai.com/api-keys> and choose
   **Create new secret key**. Name it (for example `humanfirst-demo`).
5. **Copy the key now.** It starts with `sk-` and is shown only once.

Treat the key like a password:

- Never commit it, paste it into chat, or include it in screenshots or
  logs. If it leaks, revoke it on the same page and create a new one.
- Each team member should use their own key.

### Step 3: Create the `.env` file

The key goes in a file named `.env` **inside the `backend` folder**
(next to `app.py`). The file is listed in `.gitignore`, so it is never
pushed to GitHub.

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
running. The top right of the page should read *Service online*.

### Try it

Paste this into **Triage a message**:

> My electricity was disconnected this morning and my insulin has to stay
> refrigerated. I don't know what to do.

Expected: `Critical`, classified by `OpenAI`, flagged for human review.

### Troubleshooting

| Symptom                                                    | Cause and fix                                                                                   |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Backend crashes on start with `Missing credentials`        | `.env` is missing, in the wrong folder, or misnamed. It must be `backend/.env`, containing `OPENAI_API_KEY=...`. |
| Every case shows **Local ML** and **Provisional**          | OpenAI call is failing. Check the backend terminal for `[OpenAI unavailable]`: usually an invalid key, no credit, or no internet. |
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
| `OPENAI_API_KEY` | Yes      | Key used for the primary classifier |

### Backend settings

| What                          | Where                                                    | Notes                                                                 |
|-------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------|
| OpenAI model                  | `model="gpt-5-mini"` in `classify_message()`, `openai_analyzer.py` | Swap for another model that supports structured output                |
| Classification rules          | `SYSTEM_PROMPT` in `openai_analyzer.py`                  | Defines Critical / High / Normal and the "tone is not evidence" rule  |
| Categories and teams          | `Category` enum in `openai_analyzer.py`                  | Keep in step with `ESCALATION_MAP` and the training CSV               |
| Fallback training data        | `DEFAULT_TRAINING_CSV` in `app.py`                       | A labelled CSV the Local ML model trains on at start-up               |
| Server port and host          | `app.run(...)` at the bottom of `app.py`                 | Default `0.0.0.0:5000` with `debug=True`. For a demo or any shared network, use `host="127.0.0.1"` and `debug=False` |
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
