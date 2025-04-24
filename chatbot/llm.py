# Directory: llm_chatbot_server/chatbot/
# Purpose: Manages LLM initialization and functions for language detection (DeepSeek R1) and response extraction.
# Prerequisites:
#   - Dependencies: langchain, langchain-ollama.
#   - Ollama server running with deepseek-r1 model available.
#   - chatbot/types.py must exist for AgentState type.
# Notes:
#   - Handles multilingual inputs (English, French, Arabic) as requested on April 18, 2025.
#   - Detects language without translation, keeping original input (e.g., "bonsoir" remains "bonsoir").
#   - Start Ollama: `ollama run deepseek-r1` (or configure as needed).

import re
import json
import logging
import traceback
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from chatbot.types import AgentState
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOllama(model="deepseek-r1", temperature=0.0)


def detect_language(state: AgentState, config: RunnableConfig) -> AgentState:
    """Uses LLM to detect the language of user_input (English, French, or Arabic) without translation."""
    user_input = state["user_input"].strip()
    state["original_input"] = user_input
    state["language"] = "english"

    try:
        prompt = (
            f"Given the input: '{user_input}', perform the following task:\n"
            f"1. Detect the language, which must be one of: English, French, or Arabic.\n"
            f"Output exactly in this format, with no additional text or comments:\n"
            f"**Language:** detected_language"
        )
        message = HumanMessage(content=prompt)
        response = ""
        logger.info(f"Streaming LLM response for language detection of '{user_input}':")
        for chunk in llm.stream([message]):
            chunk_content = chunk.content
            response += chunk_content
            print(chunk_content, end="", flush=True)
        print()
        # logger.info(f"Complete LLM response: {response}")

        language = extract_answer(response, "**Language:**").lower()
        valid_languages = {"english", "french", "arabic"}
        if language not in valid_languages:
            logger.warning(
                f"Invalid language detected: {language}, defaulting to English"
            )
            language = "english"
        state["language"] = language
        state["user_input"] = user_input  # Keep original input

        logger.info(
            f"Detected language: {language}, keeping original input: {user_input}"
        )

    except Exception as e:
        logger.error(f"Error in LLM language detection: {str(e)}")
        logger.error(traceback.format_exc())
        state["language"] = "english"
        state["user_input"] = user_input

    logger.info(f"State after detect_and_translate: {state}")
    return state


def extract_answer(response: str, key: str) -> str | list:
    print(key)
    """
    Extract the value associated with a key from the LLM response.
    Returns a string for most keys, or a list for **Products:**.
    """
    # logger.info(f"Raw LLM response: {response}")
    if not isinstance(response, str):
        logger.error(f"Invalid LLM response type: {type(response)}")
        return "none" if key != "**Products:**" else []

    # Check for JSON in Markdown code blocks
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            if key == "**Intent:**" and "intent" in json_data:
                valid_intents = {
                    "new_order",
                    "retrieve_order",
                    "list_products",
                    "greeting",
                    "address",
                    "update_address",
                    "report_issue",
                    "none",
                }
                intent = json_data["intent"]
                return intent if intent in valid_intents else "none"
            elif key == "**Products:**" and "products" in json_data:
                products = json_data["products"]
                return (
                    products
                    if isinstance(products, list)
                    else products.split(",") if isinstance(products, str) else []
                )
            return str(json_data.get(key, "none" if key != "**Products:**" else []))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return "none" if key != "**Products:**" else []

    if key in response:
        parts = response.split(key)
        if len(parts) > 1:
            value = parts[-1].split("**")[0].strip()
            # print(value)
            if key == "**Intent:**":
                valid_intents = {
                    "new_order",
                    "retrieve_order",
                    "list_products",
                    "greeting",
                    "address",
                    "update_address",
                    "report_issue",
                    "none",
                }
                return value.lower() if value.lower() in valid_intents else "none"
            elif key == "**Products:**":
                return (
                    [item.strip() for item in value.split(",") if item.strip()]
                    if value
                    else []
                )
            elif key == "**IssueProduct:**":
                return (
                    [item.strip() for item in value.split(",") if item.strip()]
                    if value
                    else []
                )

            elif key == "**Category:**":
                valid_category = {
                    "defective",
                    "wrong_item",
                    "missing_item",
                    "delivery",
                    "quality",
                    "quantity",
                    "packaging",
                    "other",
                }
                return value if value in valid_category else "none"
            elif key == "**Language:**":
                valid_languages = {"english", "french", "arabic"}
                return value.lower() if value.lower() in valid_languages else "none"
            elif key == "**Response:**":
                return value
            elif key == "**Address:**":
                return value

    logger.warning(
        f"Unexpected LLM response format for {key}: {response}. Defaulting to 'none' or []"
    )
