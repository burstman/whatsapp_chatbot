# server.py
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Any
from langserve import add_routes
from langchain_core.runnables import RunnableLambda, ConfigurableField
from logic.agent_state import AgentState, RunnableConfig
from logic.extarct import (
    check_relevance,
    check_product_existence,
    extract_product_items,
)
from logic.get_user_schema import get_current_user
from logic.sql import convert_nl_to_sql, execute_sql
import json

app = FastAPI(
    title="Agent Nodes API",
    version="1.0",
    description="API for testing agent nodes with LangServe",
)

check_relevance_runnable = RunnableLambda(check_relevance)
check_food_existence_runnable = RunnableLambda(check_product_existence)
get_current_user_runnable = RunnableLambda(get_current_user)


# # server.py (add this before uvicorn.run)
# def test_config(state: dict, config: RunnableConfig) -> dict:
#     return {"config_test": config["configurable"].get("test_key", "No key provided")}


# add_routes(
#     app,
#     RunnableLambda(test_config),
#     path="/test_config",
#     input_type=dict,
#     output_type=dict,
#     config_keys=["configurable"],
# )
add_routes(
    app,
    RunnableLambda(convert_nl_to_sql),
    path="/convert_nl_to_sql",
    input_type=AgentState,
    output_type=AgentState,
    config_keys=["configurable"],
)


add_routes(
    app,
    RunnableLambda(check_relevance),
    path="/check_relevance",
    input_type=AgentState,
    output_type=AgentState,
    config_keys=["configurable"],
)

add_routes(
    app,
    RunnableLambda(extract_product_items),
    path="/extract_product",
    input_type=AgentState,
    output_type=AgentState,
    config_keys=["configurable"],
)

add_routes(
    app,
    RunnableLambda(check_product_existence),
    path="/check_product_existence",
    input_type=AgentState,
    output_type=AgentState,
    config_keys=["configurable"],
)


# Function to modify the config based on the request
def inject_user_id_from_request(config: Dict[str, Any], req: Request) -> Dict[str, Any]:
    try:
        body = req._body.decode("utf-8")
        body_json = json.loads(body)
        print(f"Received request body: {body_json}")

        # Extract config and input from the request
        request_config = body_json.get("config", {})
        input_data = body_json.get("input", {})

        # Ensure config has the correct structure
        if "configurable" not in config:
            config["configurable"] = {}

        # Use the current_user_id from the request's config if available
        request_configurable = request_config.get("configurable", {})
        user_id = request_configurable.get("current_user_id")

        # Fallback to input if current_user_id is not in config
        if user_id is None and "current_user_id" in input_data:
            user_id = input_data["current_user_id"]
            print("Using current_user_id from input as fallback")

        if user_id:
            config["configurable"]["current_user_id"] = user_id
            print(f"Injected user_id {user_id} into config")

        return config
    except Exception as e:
        print(f"Error parsing request body: {e}")
        raise HTTPException(400, "Invalid request body")


add_routes(
    app,
    RunnableLambda(get_current_user),
    path="/get_current_user",
    config_keys=["configurable"],
    per_req_config_modifier=inject_user_id_from_request,
    enabled_endpoints=[
        "invoke",
        "playground",
        "input_schema",
        "output_schema",
        "config_schema",
    ],
    input_type=dict,
    output_type=AgentState,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
