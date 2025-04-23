# Directory: llm_chatbot_server/chatbot/
# Purpose: Defines the LangGraph workflow for processing user inputs and generating responses.
# Prerequisites:
#   - Dependencies: langgraph.
#   - chatbot/handlers.py for process_input, handle_list_products, handle_none, handle_greeting.
#   - chatbot/api.py for api_call.
# Notes:
#   - Routes intents to appropriate handlers based on state['intent'].
#   - Uses api_call('list_products') to populate state['products'].
#   - Ensures list_products intents are handled by handle_list_products.

from langgraph.graph import StateGraph, END
from chatbot.handlers import (
    process_input,
    handle_list_products,
    handle_none,
    handle_greeting,
    handle_new_order,
    handle_address_input,
    retrieve_order,
    handle_report_issue,
)
from chatbot.llm import detect_language
from chatbot.types import AgentState
import logging

logger = logging.getLogger(__name__)


# Define the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("detect_language", detect_language)
workflow.add_node("process_input", process_input)
workflow.add_node("handle_list_products", handle_list_products)
workflow.add_node("handle_none", handle_none)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("handle_new_order", handle_new_order)
workflow.add_node("handle_address_input", handle_address_input)
workflow.add_node("retrieve_order", retrieve_order)
workflow.add_node("handle_report_issue", handle_report_issue)


# Define routing function
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "none")
    next_step = state.get("next_step", None)
    logger.info(f"Routing intent: {intent}")
    if next_step == "await_address":
        return "handle_address_input"
    if intent == "list_products":
        return "handle_list_products"
    elif intent == "greeting":
        return "handle_greeting"
    elif intent == "none":
        return "handle_none"
    elif intent == "new_order":
        return "handle_new_order"
    elif intent == "retrieve_order":
        return "retrieve_order"
    elif intent == "report_issue":
        return "handle_report_issue"
    else:
        logger.warning(f"Unhandled intent: {intent}, routing to handle_none")
        return "handle_none"  # Default for unhandled intents


def route_next_step(state: AgentState) -> str:
    next_step = state.get("next_step", None)
    logger.info(f"Routing next_step: {next_step}")
    if next_step == "await_address":
        return "handle_address_input"
    return END


# Set entry point
workflow.set_entry_point("detect_language")

# Add edges
workflow.add_edge("detect_language", "process_input")
# workflow.add_edge("process_input", "fetch_products")
workflow.add_conditional_edges(
    "process_input",
    route_intent,
    {
        "handle_list_products": "handle_list_products",
        "handle_greeting": "handle_greeting",
        "handle_none": "handle_none",
        "handle_new_order": "handle_new_order",
        "handle_address_input": "handle_address_input",
        "retrieve_order": "retrieve_order",
        "handle_report_issue": "handle_report_issue",
    },
)

workflow.add_conditional_edges(
    "handle_new_order",
    route_next_step,
    {
        "handle_address_input": "handle_address_input",
        END: END,
    },
)

# Add edges to END
workflow.add_edge("handle_list_products", END)
workflow.add_edge("handle_none", END)
workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_address_input", END)
workflow.add_edge("retrieve_order", END)
workflow.add_edge("handle_report_issue", END)

# Compile the graph
graph = workflow.compile()

# Export the graph for use in webhook
__all__ = ["graph"]
