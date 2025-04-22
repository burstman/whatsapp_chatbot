# Directory: llm_chatbot_server/chatbot/
# Purpose: Defines shared types (AgentState) used across chatbot modules.
# Prerequisites: None.
# Notes:
#   - Provides type safety for LangGraph state management.
#   - Extend this file if new state fields are needed.

from typing import TypedDict, List


class AgentState(TypedDict):
    user_input: str
    original_input: str
    language: str
    address: str
    response: str
    intent: str
    next_step: str
    order_data: dict
    requested_items: List[str]
    issue_product: str
    phone_number: str
