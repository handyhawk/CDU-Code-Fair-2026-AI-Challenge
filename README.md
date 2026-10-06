# HumanFirst AI

An AI-assisted triage system for a government inbox. It reads incoming
messages, works out how urgent each one is, sorts them into a category
and a team to route them to, and flags the risky or uncertain ones for
a human to check — the AI assists, a human always decides.

## 1. How the system works

```
message  →  backend analyses it  →  saved as a "case"  →  frontend shows it
```

For every message, the backend works out three things:

- **Urgency** — `Critical`, `High`, or `Normal`
- **Category** — what the message is about (Housing, Health & Safety, etc.)
- **Route** — which team should actually handle it (usually matches the
  category, but a housing issue with a medical angle gets routed to
  Health & Safety instead)

It uses two signals together:

- A small set of **risk keywords** (medication dependency, children,
  loss of essential services, threats, etc.) — these describe real
  circumstances, so they're allowed to raise the urgency score.
- A **trained model** that reads the whole message for context. This
  matters because words like "urgent" or "!!!" *don't* automatically
  mean something is urgent — one of the training messages is literally
  "URGENT!!! I forgot my password 😂", labeled as routine. Tone isn't
  evidence; only the model reading the actual content is.

If the model isn't confident, or the case is Critical, it's flagged
**escalate to human** — someone should look at it before it's acted
on. Every analysed message is saved permanently (in `backend/data/cases.db`),
so nothing is lost between sessions. A human can then confirm or
override the AI's call, and that decision is saved too.

## 2. How to set up the application

You need Python 3.10 or newer.

**Start the backend** (does the analysis, keeps the data):
```bash
cd backend
pip install -r requirements.txt
Set-Content -Path .env -Value "OPENAI_API_KEY=sk-your-key-here"
python app.py
```
Leave this running — it serves on `http://localhost:5000`.

**Start the frontend** (the screen you interact with), in a *second*
terminal:
```bash
cd frontend
pip install streamlit requests
streamlit run app.py
```
This opens in your browser automatically, usually at `http://localhost:8501`.

Both need to be running at the same time for the app to work.

## 3. How to configure the main script

The main backend script is `backend/app.py`. A few things you can change:

| What                          | Where                                          | Notes                                                  |
|-------------------------------|-------------------------------------------------|---------------------------------------------------------|
| Which CSV it trains on         | `DEFAULT_TRAINING_CSV` in `app.py`               | Swap in a new labeled CSV as more cases are collected  |
| Server port                   | last line of `app.py` (`app.run(...)`)           | Default `5000`                                         |
| How cautious escalation is    | `ESCALATION_CONFIDENCE_THRESHOLD` in `urgency_analyzer.py` | Lower = fewer cases sent to a human; see that file's comment for how this number was chosen |
| Risk keywords                 | `SUBSTANTIVE_RISK_KEYWORDS` in `urgency_analyzer.py` | Add words/phrases that should count as real risk factors |
| "Sounds urgent but isn't proof" words | `ATTENTION_MARKER_WORDS` in `urgency_analyzer.py` | Words that get flagged for the reviewer but never raise the score |

On the frontend side, `frontend/app.py` has a **Backend URL** box in the
sidebar — change it there if the backend isn't running on your machine
at the default address.

To retrain the model with a new CSV without restarting anything, send
it to the backend directly:
```bash
curl -X POST http://localhost:5000/train -F "file=@path/to/new_cases.csv"
```
