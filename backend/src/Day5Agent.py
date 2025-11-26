import json
import os
import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    MetricsCollectedEvent,
    metrics,
)

from livekit.plugins import silero, google, murf, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("day5_sdr")
load_dotenv(".env.local")

# ------------------- FILE PATHS -------------------

BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, "company_data", "swiggy_faq.json")
LEADS_PATH = os.path.join(BASE_DIR, "leads", "day5_leads.json")

os.makedirs(os.path.dirname(LEADS_PATH), exist_ok=True)
if not os.path.exists(LEADS_PATH):
    with open(LEADS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# ------------------- FAQ LOADING -------------------

def load_faq():
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

FAQ = load_faq()

def match_faq(user_query: str):
    q = user_query.lower()
    for item in FAQ["faqs"]:
        if any(token in q for token in item["question"].lower().split()):
            return item["answer"]
    return None

# ------------------- LEAD SAVE -------------------

def save_lead(lead):
    with open(LEADS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(lead)
    with open(LEADS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ------------------- SDR AGENT -------------------

class SwiggySDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly SDR for Swiggy.
Your goals:
1. Greet warmly.
2. Answer ONLY using the FAQ.
3. Collect lead details:
   - name
   - company
   - email
   - role
   - use_case
   - team_size
   - timeline
4. When the user says “that's all”, “thanks”, “bye”, summarize and save the lead.

Keep responses short and friendly.
"""
        )

        self.lead_fields = [
            "name", "company", "email",
            "role", "use_case",
            "team_size", "timeline"
        ]

    async def on_message(self, message, session):
        user_text = message.text.strip()
        lt = user_text.lower()

        # End command
        if any(x in lt for x in ["that's all", "thanks", "bye", "i'm done"]):
            summary = self._summary(session.userdata["lead"])
            await session.send_text(summary)
            save_lead(session.userdata["lead"])
            return

        # FAQ response
        faq_answer = match_faq(user_text)
        if faq_answer:
            await session.send_text(faq_answer)

        # Lead filling
        lead = session.userdata["lead"]

        for field in self.lead_fields:
            if lead[field] is None:
                lead[field] = user_text
                await session.send_text(f"Great, noted your {field}.")
                next_index = self.lead_fields.index(field) + 1

                if next_index < len(self.lead_fields):
                    await session.send_text(
                        f"Could you tell me your {self.lead_fields[next_index]}?"
                    )
                else:
                    await session.send_text("Thanks! Say 'That's all' to finish.")
                return

        await session.send_text("Thanks! Feel free to ask anything else!")

    def _summary(self, lead):
        return (
            "Here’s your summary:\n\n"
            f"Name: {lead['name']}\n"
            f"Company: {lead['company']}\n"
            f"Email: {lead['email']}\n"
            f"Role: {lead['role']}\n"
            f"Use Case: {lead['use_case']}\n"
            f"Team Size: {lead['team_size']}\n"
            f"Timeline: {lead['timeline']}\n\n"
            "Thank you for speaking with Swiggy SDR!"
        )

# ------------------- PREWARM -------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Day5 SDR Agent prewarmed.")

# ------------------- ENTRYPOINT -------------------

async def entrypoint(ctx: JobContext):

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

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    # Initialize userdata
    session.userdata = {}
    session.userdata["lead"] = {
        "name": None,
        "company": None,
        "email": None,
        "role": None,
        "use_case": None,
        "team_size": None,
        "timeline": None,
    }

    await session.start(
        agent=SwiggySDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# ------------------- MAIN RUNNER -------------------

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
