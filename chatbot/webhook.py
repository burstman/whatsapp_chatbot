# Directory: llm_chatbot_server/chatbot/
# Purpose: Defines the FastAPI app, WhatsApp webhook endpoint, and health check for Twilio integration.
# Prerequisites:
#   - Dependencies: fastapi, twilio, langserve, langchain.
#   - chatbot/db.py for SessionLocal, User.
#   - chatbot/llm.py for llm, extract_answer.
#   - chatbot/graph.py for graph.
#   - chatbot/types.py for AgentState.
#   - TWILIO_AUTH_TOKEN environment variable.
# Notes:
#   - Handles WhatsApp messages via /whatsapp/webhook.
#   - Validates Twilio signatures using TWILIO_AUTH_TOKEN directly.
#   - Sends responses via Twilio WhatsApp API.
#   - Fixed AttributeError by removing messaging_service_sid reference.

import logging
import os
import traceback
from fastapi import FastAPI, Request, HTTPException, Form
from langserve import add_routes
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from twilio.base.exceptions import TwilioRestException
from chatbot.db import SessionLocal, User
from chatbot.types import AgentState
from chatbot.llm import llm, extract_answer
from chatbot.graph import graph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Webhook Agent")

# Initialize conversation states
conversation_states = {}


# Twilio client (initialized externally)
def set_twilio_client(client: Client):
    global twilio_client
    twilio_client = client


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
    ProfileName: str = Form(default="Unknown User"),
):
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not auth_token:
        logger.error("TWILIO_AUTH_TOKEN not set")
        raise HTTPException(status_code=500, detail="Twilio configuration error")
    validator = RequestValidator(auth_token)
    url = str(request.url)
    signature = request.headers.get("X-Twilio-Signature", "")
    form_data = await request.form()
    # logger.info(f"Webhook URL: {url}")
    # logger.info(f"Twilio Signature: {signature}")
    # logger.info(f"Form Data: {form_data}")
    if not validator.validate(url, form_data, signature):
        logger.warning(f"Invalid Twilio signature from {From}")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    logger.info(f"Received WhatsApp message from {From}: {Body}")

    try:

        phone_number = From
        name = ProfileName

        with SessionLocal() as session:
            try:
                converty_phone = phone_number.replace("whatsapp:+216", "")
                user = (
                    session.query(User).filter_by(phone_number=converty_phone).first()
                )
                if not user:
                    user = User(
                        phone_number=converty_phone, name=name, address="unknown"
                    )
                    session.add(user)
                    session.commit()
                    logger.info(f"Added new user {converty_phone} with name {name}")
            except Exception as e:
                session.rollback()
                logger.error(f"Error managing user {converty_phone}: {e}")
                raise

        if phone_number not in conversation_states:
            conversation_states[phone_number] = AgentState(
                user_input="",
                original_input=None,
                language="english",
                address="unknown",
                response="",
                intent=None,
                next_step=None,
                order_data=None,
                requested_items=None,
                issue_product=None,
            )

        state = conversation_states[phone_number].copy()
        state["user_input"] = Body.strip()

        config = RunnableConfig(
            configurable={"name": name, "phone_number": phone_number}
        )

        logger.info(f"Initial state before invoke: {state}, Config: {config}")
        result = graph.invoke(state, config=config)
        logger.info(f"Result from graph.invoke: type={type(result)}, value={result}")
        state.update(result)
        conversation_states[phone_number] = state

        response_message = state.get(
            "response",
            f"Sorry, I couldn’t figure that out—how can I help? ({state['language']})",
        )
        logger.info(f"Generated response_message: {response_message}")

        if "No products found" in response_message:
            prompt = (
                f"Generate a message in {state['language']} suggesting the user try products like 'Presse Agrume Silver Crest' or 'Solar Interaction Wall Lamp', or ask 'What do you have?'. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** suggestion_message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            response_message += "\n" + extract_answer(response, "**Response:**")

        try:
            confirmation_message = twilio_client.messages.create(
                from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
                body=response_message,
                to=phone_number,
            )
            logger.info(
                f"Sent confirmation message to {phone_number}. Message SID: {confirmation_message.sid}"
            )
        except TwilioRestException as e:
            logger.error(
                f"Twilio API error sending message to {phone_number}: {str(e)}"
            )
            raise

    except Exception as e:
        logger.error(f"Error processing WhatsApp message from {From}: {str(e)}")
        logger.error(traceback.format_exc())
        prompt = (
            f"Generate a friendly error message in {state.get('language', 'english')} saying something went wrong and suggesting to try again. "
            f"Output exactly in this format:\n"
            f"**Response:** error_message"
        )
        message = HumanMessage(content=prompt)
        response = llm.invoke([message]).content
        response_message = extract_answer(response, "**Response:**")
        try:
            confirmation_message = twilio_client.messages.create(
                from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
                body=response_message,
                to=phone_number,
            )
            logger.info(
                f"Sent error message to {phone_number}. Message SID: {confirmation_message.sid}"
            )
        except TwilioRestException as e:
            logger.error(
                f"Twilio API error sending error message to {phone_number}: {str(e)}"
            )
            raise

    return {"status": "message sent"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}


add_routes(app, graph, path="/agent")
