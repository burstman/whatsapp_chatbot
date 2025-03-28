from .agent_state import AgentState, RunnableConfig
from .get_user_schema import get_database_schema
from .table_db_logic import Engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from sqlalchemy import text
import re


def convert_nl_to_sql(state: dict, config: RunnableConfig) -> AgentState:
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected state to be a dictionary in check_relevance , but got {type(state).__name__}"
        )
    state_obj = AgentState(**state)
    question = state_obj.question
    current_user = state_obj.current_user
    product_exists = state_obj.product_exists
    corrected_product_items = state_obj.corrected_product_items
    sql_error = state_obj.sql_error
    sql_error_message = state_obj.sql_error_message
    attempts = state_obj.attempts

    schema = get_database_schema(Engine)
    print(f"Converting question to SQL for user '{current_user}', question: {question}")
    print(
        f"Product exists: {product_exists}, Corrected Product items: {corrected_product_items}"
    )
    print(
        f"SQL Error: {sql_error}, Error Message: '{sql_error_message}', Attempts: {attempts}"
    )

    if not product_exists or not corrected_product_items:
        print("No valid product items found, skipping SQL generation.")
        return AgentState(**state_obj.model_dump() | {"sql_query": None})

    MAX_ATTEMPTS = 3
    if attempts >= MAX_ATTEMPTS:
        print(
            f"Maximum retry attempts ({MAX_ATTEMPTS}) reached. Aborting SQL generation."
        )
        return AgentState(
            **state.model_dump()
            | {
                "sql_query": None,
                "sql_error": True,
                "sql_error_message": f"Failed to generate a valid SQL query after {MAX_ATTEMPTS} attempts.",
            }
        )

    system_base = (
        "You are an assistant that converts natural language questions into SQL queries based on the following schema:\n\n"
        f"{schema}\n\n"
        f"The current user is '{current_user}'. The product items for this query are {corrected_product_items}. Use these values directly in the SQL query without placeholders.\n"
        "For INSERT queries to create orders (e.g., 'Make a new order for X'), generate an INSERT statement using a JOIN approach: "
        "`INSERT INTO orders (user_id, product_id) SELECT u.id, p.id FROM users u JOIN product p ON p.name = '<corrected_product_item>' WHERE u.name = '<current_user>'`, replacing '<current_user>' with the actual user name and '<corrected_product_item>' with the actual product name. Note: INSERT statements do not return rows, only modify the database.\n"
        "For multiple distinct product items in the question (e.g., 'Make a new order for X and Y'), use: "
        "`INSERT INTO orders (user_id, product_id) SELECT u.id, p.id FROM users u JOIN product p ON p.name IN ('<product1>', '<product2>') WHERE u.name = '<current_user>'`, listing all product items in the IN clause.\n"
        "Provide ONE query for all product items, wrapped in ```sql ... ``` tags.\n"
        "For queries asking about available product or the menu (e.g., 'What’s on the menu?'), use: `SELECT name, price FROM product`.\n"
        "For queries about a user's orders (e.g., 'Show me my orders'), use: "
        "`SELECT o.id, p.name, p.price FROM orders o JOIN product p ON o.product_id = p.id WHERE o.user_id = (SELECT id FROM users WHERE name = '<current_user>')`.\n"
        "Provide the SQL queries wrapped in ```sql ... ``` tags, with no explanations.\n"
        "Example: For 'Make a new order for Boite lunch Box and Presse Agrume Silver Crest with user 'Charlie' and product_items ['Boite lunch Box', 'Presse Agrume Silver Crest'], output: "
        "```sql\nINSERT INTO orders (user_id, product_id) SELECT u.id, p.id FROM users u JOIN product p ON p.name IN ('Boite lunch Box', 'Presse Agrume Silver Crest') WHERE u.name = 'Charlie'\n```"
    )

    if sql_error and sql_error_message:
        print(f"sql_error: {sql_error}, sql_error_message: {sql_error_message}")
        if state_obj.row_count == 0:
            print(
                "SQL error is due to execution expecting rows from INSERT. Marking as non-retryable."
            )
            return AgentState(
                **state.model_dump()
                | {
                    "sql_query": state.sql_query,
                    "sql_error": True,
                    "sql_error_message": "INSERT executed successfully but does not return rows as expected by the executor.",
                }
            )
        system = (
            f"{system_base}\n\n"
            f"The previous SQL query failed with the error: '{sql_error_message}'. "
            "Correct the query to resolve this error, adhering to the schema and using the current user and corrected product items directly."
        )
        print(
            "Previous SQL query failed. Attempting to correct the query with the LLM."
        )
    else:
        system = system_base

    convert_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", f"Question: {question}")]
    )
    llm = ChatOllama(model="deepseek-r1", temperature=0, stream=True)
    chain = convert_prompt | llm | StrOutputParser()

    full_sql_query = ""
    print("Streaming SQL generation: ", end="")
    for chunk in chain.stream({"question": question}):
        full_sql_query += chunk
        print(chunk, end="", flush=True)
    print("")
    full_sql_query = full_sql_query.strip()

    full_sql_query = re.sub(
        r"<think>.*?</think>", "", full_sql_query, flags=re.DOTALL
    ).strip()
    print(f"SQL query after removing <think> tags: '{full_sql_query}'")

    sql_queries = []
    sql_blocks = re.findall(r"```sql\s*(.+?)\s*```", full_sql_query, re.DOTALL)
    for block in sql_blocks:
        query = block.strip()
        if query:
            sql_queries.append(query)
            print(f"Extracted SQL query: '{query}'")

    if not sql_queries:
        raise ValueError(f"No SQL queries found in: '{full_sql_query}'")

    new_attempts = (
        attempts + 1
        if sql_error and "does not return rows" not in sql_error_message.lower()
        else 0
    )
    return AgentState(
        **state_obj.model_dump() | {"sql_query": sql_queries, "attempts": new_attempts}
    )


def execute_sql(state: AgentState, config: RunnableConfig) -> AgentState:
    print(f"Executing SQL: {state.sql_query}")
    with Engine.connect() as connection:
        try:
            query = state.sql_query[0] if state.sql_query else ""
            if not query:
                return AgentState(
                    **state.model_dump()
                    | {"sql_error": True, "sql_error_message": "No SQL query provided"}
                )
            result = connection.execute(text(query))
            connection.commit()
            row_count = result.rowcount  # Number of rows inserted
            print(f"SQL request executed successfully, {row_count} row(s) affected")
            return AgentState(
                **state.model_dump()
                | {
                    "query_rows": None,  # No rows to return for INSERT
                    "sql_error": False,
                    "sql_error_message": None,
                    "row_count": row_count,  # Optional: track affected rows
                }
            )
        except Exception as e:
            connection.rollback()
            return AgentState(
                **state.model_dump() | {"sql_error": True, "sql_error_message": str(e)}
            )
