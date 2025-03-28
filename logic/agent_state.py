import os
from dotenv import load_dotenv
import ast
from langchain_core.runnables.config import RunnableConfig
import logging
from typing_extensions import TypedDict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text, inspect
from langgraph.graph import StateGraph, END
import re
from .table_db_logic import User, Product, Order, SessionLocal, Engine


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("langchain_ollama").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


class AgentState(BaseModel):
    question: str
    sql_query: List[str] = Field(default=[])
    query_result: str = Field(default="")
    query_rows: List = Field(default=[])
    current_user: str = Field(default="Unknown_user")
    current_user_phone: int = Field(default=00000000)
    attempts: int = Field(default=0)
    relevance: str = Field(default="")
    sql_error: bool = Field(default=False)
    sql_error_message: str = Field(default="")
    row_count: int = Field(default=0)
    product_item: str = Field(default="")
    product_exists: bool = Field(default=False)
    corrected_product_items: List[str] = Field(default_factory=list)


@field_validator("sql_query")
def ensure_sql_query_is_list(cls, v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return v
    raise ValueError("sql_query must be a string or list of strings")


@field_validator("corrected_food_items")
def validate_food_items(cls, v):
    if not all(isinstance(item, str) for item in v):
        raise ValueError("All corrected_food_items must be strings")
    return v


# def get_database_schema(engine=Engine):
#     inspector = inspect(engine)
#     schema = ""
#     for table_name in inspector.get_table_names():
#         schema += f"Table: {table_name}\n"
#         for column in inspector.get_columns(table_name):
#             col_name = column["name"]
#             col_type = str(column["type"])
#             if column.get("primary_key"):
#                 col_type += ", Primary Key"
#             if column.get("foreign_keys"):
#                 fk = list(column["foreign_keys"])[0]
#                 col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
#             schema += f"- {col_name}: {col_type}\n"
#         schema += "\n"
#     print("Retrieved database schema.")
#     return schema


# def get_current_user(state: AgentState, config: RunnableConfig) -> AgentState:
#     print("Retrieving the current user based on user ID.")
#     user_id = config["configurable"].get("current_user_id", None)
#     if not user_id:
#         print("No user ID provided in the configuration.")
#         return AgentState(**state.model_dump() | {"current_user": "User not found"})

#     session = SessionLocal()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if user:
#             current_user = user.name
#             print(f"Current user set to: {current_user}")
#             return AgentState(**state.model_dump() | {"current_user": current_user})
#         else:
#             print("User not found in the database.")
#             return AgentState(**state.model_dump() | {"current_user": "User not found"})
#     except Exception as e:
#         print(f"Error retrieving user: {str(e)}")
#         return AgentState(
#             **state.model_dump() | {"current_user": "Error retrieving user"}
#         )
#     finally:
#         session.close()


# def check_relevance(state: AgentState, config: RunnableConfig) -> AgentState:
#     question = state.question
#     current_user = state.current_user

#     schema = get_database_schema(engine)
#     print(f"Checking relevance of the question: {question}")

#     system = (
#         "You are an assistant that determines if a user's question is relevant to a database schema.\n\n"
#         f"Schema:\n{schema}\n\n"
#         "The database includes tables for users, food items, and orders. The users table stores user information (id, name, age, email). "
#         "The food table stores food items (id, name, price). The orders table stores orders (id, food_id, user_id).\n"
#         "- If the question involves ordering food, querying the menu, or retrieving user orders, it is relevant. Output 'relevant'.\n"
#         "- Everythink related about food and restaurent are relevant.\n"
#         "- If the question is unrelated (e.g., about the weather, general knowledge), it is irrelevant. Output 'irrelevant'.\n"
#         "Provide the result as plain text, with NO explanations, quotes, or extra text.\n"
#         "Examples:\n"
#         "what kind you restaurent can prepare?\n"
#         "- 'Make a new order for Pizza' -> relevant\n"
#         "- 'What is the weather like?' -> irrelevant"
#     )

#     relevance_prompt = ChatPromptTemplate.from_messages(
#         [("system", system), ("human", f"Question: {question}")]
#     )
#     llm = ChatOllama(model="mistral", temperature=0)
#     chain = relevance_prompt | llm | StrOutputParser()

#     full_output = ""
#     print("Streaming relevance check (via LLM): ", end="")
#     for chunk in chain.stream({"question": question}):
#         full_output += chunk
#         print(chunk, end="", flush=True)
#     print("")

#     full_output = re.sub(
#         r"<think>.*?</think>", "", full_output, flags=re.DOTALL
#     ).strip()
#     relevance = full_output.strip()
#     print(f"Relevance determined: {relevance}")
#     data = state.model_dump()
#     data["relevance"] = relevance
#     return AgentState(**data)


# def extract_food_items(question: str) -> tuple[str, list[str]]:
#     """Extract food items from the question. Returns (raw_food_string, food_items)."""
#     system = """You are an assistant that extracts specific food items from a question.
# For questions like 'Do you make [cuisine] Food like [food1] and [food2]?' or 'Create a new order for [food1] and [food2]', extract only the core food names, excluding articles ('a', 'an', 'the') and prepositions ('to', 'of'), and return them as a Python tuple string.
# For generic questions about the menu (e.g., 'What’s on the menu?', 'What do you have in the menu?'), return '("", [])'.
# Output ONLY a single-line Python tuple string with two elements:
# - First element: a comma-separated list of core food names (e.g., 'food1, food2') or '' if no specific items.
# - Second element: a list of individual core food names (e.g., ['food1', 'food2']) or [] if no specific items.
# Use single quotes, no extra spaces around commas, and NO additional text, explanations, or tags. Examples:
# - 'Create a new order for couscous and Spaghetti Carbonara' -> '("couscous, Spaghetti Carbonara", ["couscous", "Spaghetti Carbonara"])'
# - 'Make a new order for pizza margaritta, spaghetty carbonara and a hamburger' -> '("pizza margaritta, spaghetty carbonara, hamburger", ["pizza margaritta", "spaghetty carbonara", "hamburger"])'
# - 'Do you make Italian Food like a pizza and pasta to go?' -> '("pizza, pasta", ["pizza", "pasta"])'
# - 'What’s on the menu?' -> '("", [])'
# DO NOT add any interpretation or explation!
# """

#     extract_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", f"Extract food items from this question: {question}"),
#         ]
#     )
#     llm = ChatOllama(model="deepseek-r1", temperature=0)
#     chain = extract_prompt | llm | StrOutputParser()

#     result = ""
#     print("Generating...: ", end="")
#     for chunk in chain.stream({"question": question}):
#         result += chunk
#         print(chunk, end="", flush=True)
#     print("")

#     result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
#     print(f"LLM extract result: '{result}'")

#     try:
#         parsed_result = ast.literal_eval(result)
#         if not isinstance(parsed_result, tuple) or len(parsed_result) != 2:
#             raise ValueError("Result is not a valid tuple with 2 elements")
#         raw_food_string, food_items = parsed_result
#         if not isinstance(raw_food_string, str) or not isinstance(food_items, list):
#             raise ValueError("Tuple elements are not (str, list)")
#     except (ValueError, SyntaxError) as e:
#         print(f"Failed to parse tuple result: {e}. Falling back to empty values.")
#         raw_food_string = ""
#         food_items = []

#     return raw_food_string, food_items
