from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from dotenv import load_dotenv
import os
import sys
from time import sleep

from rag_engine import (
    normalize_text,
    get_answer,
    index_new_message
)

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
MONITORED_CHANNELS = os.getenv("MONITORED_CHANNELS", "").split(",")

print("🔧 SLACK_BOT_TOKEN:", SLACK_BOT_TOKEN)
print("🔧 SLACK_APP_TOKEN:", SLACK_APP_TOKEN)
print("🔧 MONITORED_CHANNELS:", MONITORED_CHANNELS)



print("✅ Slack listener started...")
print(f"Monitoring channels: {MONITORED_CHANNELS}")

app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

@app.event("message")
def handle_message_events(body, logger):
    event = body.get("event", {})
    channel = event.get("channel")
    text = event.get("text")
    user = event.get("user")
    subtype = event.get("subtype")

    if user is None or event.get("bot_id") or subtype is not None or not text:
        return

    try:
        info = client.conversations_info(channel=channel)
        channel_name = info['channel']['name']
        print(f"📥 Message in #{channel_name}: {text}")

        if channel_name in MONITORED_CHANNELS:
            if text.strip().startswith("?"):
                query = text.strip()[1:].strip()
                print(f"🔍 Detected query: {query}")
                answer = get_answer(query)
                response = f"🤖 *Answer:*\n{answer}"
                client.chat_postMessage(channel=channel, text=response)
            else:
                if index_new_message(text):
                    print(f"📦 Indexed new message.")

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
