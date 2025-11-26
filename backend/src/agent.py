#smarter way to print logs with additional information.
import logging
#import env variables from a .env file into the environment.
from dotenv import load_dotenv
#livekit library imports for building conversational agents.
#livekit provides a room for agent and human to talk in real time without pressing buttons.
from Day6Agent import FraudAgent


from livekit.agents import (
    Agent,#tells the behaviour of the agent.
    AgentSession,#connect the agent to the room and manage the conversation,connects tts,sst,llm,vad etc also.
    JobContext,
    JobProcess,
    MetricsCollectedEvent,#saves info like api calls etc
    RoomInputOptions,#
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins import murfai as murf
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tools.orderTool import save_order_to_json  # Import tool

logger = logging.getLogger("agent")
#logger object to log messages with the name "agent".
load_dotenv(".env.local")
#load environment variables from the .env.local file.

#Agent is a base class of Livekit that represents the blue print of the agent.
#We create a subclass of Agent called Assistant to define the specific behavior and instructions for our barista agent.
#instructions are passed to the super class constructor to set up the agent's behavior.
#system prompt (that guides the agent's behavior during the conversation)=instructions here
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly barista named Ram at a coffee shop in India. 
Your job is to take voice-based coffee orders from customers.

Maintain and gradually fill this order object as you talk to the customer:
{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}

Ask follow-up questions until **all fields are completely filled**.

When the order is fully complete:
1. Summarize the final order to the customer  
2. Then **call the function `save_order_to_json` and pass the order object**  
   Example: save_order_to_json(orderObject)

Do NOT ask the user to confirm saving. Just save automatically when all fields are filled.

Think step by step and be conversational.
""",
        )

#before starting the agent,load the vad(Voice Activity Detection) model into userdata.
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

#create the entrypoint function where the agent session is created and started.
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create AgentSession and attach tools here ✔
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        tools=[save_order_to_json],   # 👈 Correct location
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start agent (no tools here) ✔
    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))