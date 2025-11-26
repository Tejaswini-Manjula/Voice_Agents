import os
import json
import sqlite3
import logging
from typing import Optional
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    RoomInputOptions,
    cli,
    tokenize,
)

from livekit.plugins import google, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import murf      # your correct Murf import

logger = logging.getLogger("fraud-agent")
load_dotenv(".env.local")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "shared-data", "day6_fraud_cases.db")


# -------------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT,
            securityIdentifier TEXT,
            cardEnding TEXT,
            transactionAmount TEXT,
            merchantName TEXT,
            location TEXT,
            transactionTime TEXT,
            transactionCategory TEXT,
            transactionSource TEXT,
            securityQuestion TEXT,
            securityAnswer TEXT,
            status TEXT,
            outcomeNote TEXT,
            raw_json TEXT
        )
    """)

    c.execute("SELECT COUNT(*) FROM fraud_cases")
    if c.fetchone()[0] == 0:
        samples = [
            {
                "userName": "John",
                "securityIdentifier": "JHN-1001",
                "cardEnding": "**** 4242",
                "transactionAmount": "₹18,499",
                "merchantName": "ABC Industries",
                "location": "Bengaluru",
                "transactionTime": "2025-11-25 18:32",
                "transactionCategory": "e-commerce",
                "transactionSource": "alibaba.com",
                "securityQuestion": "What is your favorite pet's name?",
                "securityAnswer": "fluffy",
                "status": "pending_review",
            }
        ]

        for s in samples:
            c.execute("""
                INSERT INTO fraud_cases
                (userName, securityIdentifier, cardEnding, transactionAmount, merchantName,
                 location, transactionTime, transactionCategory, transactionSource,
                 securityQuestion, securityAnswer, status, outcomeNote, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                s["userName"], s["securityIdentifier"], s["cardEnding"],
                s["transactionAmount"], s["merchantName"], s["location"],
                s["transactionTime"], s["transactionCategory"], s["transactionSource"],
                s["securityQuestion"], s["securityAnswer"], s["status"], "",
                json.dumps(s),
            ))

        conn.commit()

    conn.close()


def load_case(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM fraud_cases WHERE userName=?", (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    keys = [
        "id","userName","securityIdentifier","cardEnding","transactionAmount",
        "merchantName","location","transactionTime","transactionCategory",
        "transactionSource","securityQuestion","securityAnswer","status",
        "outcomeNote","raw_json"
    ]
    return dict(zip(keys, row))


def update_case(case_id, status, note):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE fraud_cases SET status=?, outcomeNote=? WHERE id=?",
        (status, note, case_id)
    )
    conn.commit()
    conn.close()


# -------------------------------------------------------------------------
# AGENT BEHAVIOR
# -------------------------------------------------------------------------
class FraudAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are the Axis Bank Fraud Monitoring Unit.
Speak clearly and calmly.
Never ask for PIN, OTP, CVV, passwords, or sensitive info.
Verify identity ONLY with the non-sensitive security question.
"""
        )

        self.intro = (
            "Hello, this is the Fraud Monitoring Unit calling on behalf of Axis Bank. "
            "We detected a suspicious transaction on your account. "
            "Before we proceed, I will verify one non-sensitive detail."
        )

    # LIVEKIT SPEECH MODE — NOT console mode
    async def on_session_start(self, session: AgentSession):
        await session.say(self.intro)

        name = (await session.ask("May I know your first name?")).strip()
        case = load_case(name)

        if not case:
            await session.say(
                "I could not find any pending fraud review. "
                "Please contact Axis Bank support. Ending the call."
            )
            return

        answer = (await session.ask(case["securityQuestion"])).strip().lower()
        expected = case["securityAnswer"].lower()

        if answer != expected:
            await session.say(
                "That does not match our records. For security reasons, I must end this call."
            )
            update_case(case["id"], "verification_failed", "Bad security answer")
            return

        # Transaction details
        await session.say(
            f"Thank you. A transaction at {case['merchantName']} for {case['transactionAmount']} "
            f"on your card ending {case['cardEnding']} was flagged. "
            f"It occurred on {case['transactionTime']} in {case['location']}."
        )

        txn = (await session.ask("Did you make this transaction? Yes or no?")).strip().lower()

        if txn in ("yes", "y"):
            update_case(case["id"], "confirmed_safe", "User confirmed transaction")
            await session.say("Thank you. We will mark it as safe. Have a good day.")
            return

        if txn in ("no", "n"):
            update_case(case["id"], "confirmed_fraud", "User reported fraud")
            await session.say(
                "Thank you. We have marked it as fraudulent and placed a temporary block on your card. "
                "Axis Bank support will contact you shortly."
            )
            return

        update_case(case["id"], "verification_failed", "Unclear yes/no")
        await session.say("I cannot continue without a clear yes or no. Ending the call.")


# -------------------------------------------------------------------------
# WORKER ENTRYPOINT — EXACTLY LIKE DAY 5
# -------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(   # ✔️ FIXED — Murf TTS works on your version
            model="matthew",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        tools=[],
        preemptive_generation=True,
    )

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# -------------------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
