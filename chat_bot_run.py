# Directory: llm_chatbot_server/
# Purpose: Main entry point for the chatbot server. Initializes dependencies, sets up the FastAPI server, and creates database tables.
# Prerequisites:
#   - Environment variables in .env (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT).
#   - Dependencies: fastapi, uvicorn, twilio, sqlalchemy, python-dotenv (install via requirements.txt).
#   - chatbot/ package must exist in llm_chatbot_server/chatbot/ with all modules (__init__.py, db.py, llm.py, handlers.py, api.py, webhook.py, graph.py, types.py).
# Notes:
#   - Run this file to start the server: `python chat_bot_run.py`.
#   - Creates database tables on startup using SQLAlchemy.

import os
import logging
from fastapi import FastAPI
import uvicorn
from twilio.rest import Client
from chatbot.db import engine, SessionLocal, Base
from chatbot.webhook import (
    app,
    conversation_states,
    set_twilio_client,
)  # Add set_twilio_client
from chatbot.graph import graph

from dotenv import load_dotenv

# Load environment variables from .env file
print("env:", load_dotenv())

# Environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

# Validate environment variables
for var, name in [
    (TWILIO_ACCOUNT_SID, "TWILIO_ACCOUNT_SID"),
    (TWILIO_AUTH_TOKEN, "TWILIO_AUTH_TOKEN"),
    (TWILIO_WHATSAPP_NUMBER, "TWILIO_WHATSAPP_NUMBER"),
    (DB_HOST, "DB_HOST"),
    (DB_NAME, "DB_NAME"),
    (DB_USER, "DB_USER"),
    (DB_PASSWORD, "DB_PASSWORD"),
]:
    if not var:
        raise ValueError(f"{name} environment variable is not set")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Twilio client
try:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {str(e)}")
    raise

# Set Twilio client for webhook
try:
    set_twilio_client(client)
    logger.info("Twilio client set for webhook")
except Exception as e:
    logger.error(f"Failed to set Twilio client for webhook: {str(e)}")
    raise

if __name__ == "__main__":
    # Create database tables
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
