import re
import ast
import json
from .agent_state import (
    AgentState,
    RunnableConfig,
    ChatPromptTemplate,
    ChatOllama,
    StrOutputParser,
)
from .get_user_schema import get_database_schema, SessionLocal
from .table_db_logic import User, Product, Order, Engine


def check_relevance(state: dict, config: RunnableConfig) -> AgentState:
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected state to be a dictionary in check_relevance , but got {type(state).__name__}"
        )
    state_obj = AgentState(**state)
    question = state_obj.question
    # current_user = state_obj.current_user

    schema = get_database_schema(Engine)
    print(f"Checking relevance of the question: {question}")

    system = (
        "You are an assistant that determines if a user's question is relevant to a database schema.\n\n"
        f"Schema:\n{schema}\n\n"
        "The database includes tables for users, products, and orders. The users table stores user information (id, name, phone, city, address, email). "
        "The products table stores product items (id, name, price, category). The orders table stores orders (id, product_id, user_id).\n"
        "- If the question involves ordering products, querying the catalog, or retrieving user orders, it is relevant. Output 'relevant'.\n"
        "- Everything related to products and e-commerce is relevant.\n"
        "- If the question is unrelated (e.g., about the weather, general knowledge), it is irrelevant. Output 'irrelevant'.\n"
        "Provide the result as plain text, with NO explanations, quotes, or extra text.\n"
        "Examples:\n"
        "- 'Make a new order for Boite Lunch Box' -> relevant\n"
        "- 'What is the weather like?' -> irrelevant"
    )
    relevance_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", f"Question: {question}")]
    )
    llm = ChatOllama(model="mistral", temperature=0)
    chain = relevance_prompt | llm | StrOutputParser()

    full_output = ""
    # print("Streaming relevance check (via LLM): ", end="")
    # for chunk in chain.stream({"question": question}):
    #     full_output += chunk
    #     print(chunk, end="", flush=True)
    # print("")
    full_output = chain.invoke({"question": question})

    full_output = re.sub(
        r"<think>.*?</think>", "", full_output, flags=re.DOTALL
    ).strip()
    relevance = full_output.strip()
    print(f"Relevance determined: {relevance}")
    data = state_obj.model_dump()
    data["relevance"] = relevance
    return AgentState(**data)


def extract_product_items(state: dict, config: RunnableConfig) -> AgentState:
    """Extract product items from the question and update the AgentState with raw_product_string only."""
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected state to be a dictionary in extract_product_items, but got {type(state).__name__}"
        )
    state_obj = AgentState(**state)
    question = state_obj.question

    system = """You are an assistant that extracts specific e-commerce product items from a user's question, focusing on food items when relevant. 
For questions like 'Do you have [product] Food like [product1] and [product2]?' or 'Create a new order for [product1] and [product2]', extract only the core product names, excluding articles ('a', 'an', 'the') and prepositions ('to', 'of'), and return them as a comma-separated string.
For generic questions (e.g., 'What do you sell?', 'What do you have?'), return an empty string.
Output ONLY a single-line JSON object with one field:
- "raw_product_string": a comma-separated list of core product names (e.g., "product1, product2") or "" if no specific items.
Use double quotes for strings, no extra spaces around commas, and NO additional text, explanations, or tags. Examples:
- "Create a new order for Generic Boîte Lunch Box and Presse Agrume Silver Crest" -> {{"raw_product_string": "Generic Boîte Lunch Box, Presse Agrume Silver Crest"}}
- "Make a new order for solar interaction wall lamp, Generic Boîte Lunch Box and a Presse Agrume Silver Crest" -> {{"raw_product_string": "solar interaction wall lamp, Generic Boîte Lunch Box, Presse Agrume Silver Crest"}}
- "Make a new order for pizza margaritta, spaghetty carbonara and a lazania" -> {{"raw_product_string": "pizza margaritta, spaghetty carbonara, lazania"}}
- "What do you sell?" -> {{"raw_product_string": ""}}
DO NOT add any interpretation or explanation!
"""

    extract_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"Extract product items from this question: {question}"),
        ]
    )
    llm = ChatOllama(model="deepseek-r1", temperature=0)
    chain = extract_prompt | llm | StrOutputParser()

    result = ""
    print("Generating...: ", end="")
    for chunk in chain.stream({"question": question}):
        result += chunk
        print(chunk, end="", flush=True)
    print("")

    # Remove any <think> tags or other unwanted content
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    print(f"LLM extract result: '{result}'")

    try:
        # Parse the result as JSON
        parsed_result = json.loads(result)
        raw_product_string = parsed_result.get("raw_product_string", "")
        if not isinstance(raw_product_string, str):
            raise ValueError("raw_product_string is not a string")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Failed to parse JSON result: {e}. Falling back to empty values.")
        raw_product_string = ""

    # Update the AgentState with raw_product_string
    print(f"Extracted raw product string: '{raw_product_string}'")
    return AgentState(**state_obj.model_dump() | {"product_item": raw_product_string})


def check_product_existence(state: dict, config: RunnableConfig) -> AgentState:
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected state to be a dictionary in check_product_existence, but got {type(state).__name__}"
        )
    state = AgentState(**state)

    question = state.question
    current_user = state.current_user

    print(f"Checking product existence for user '{current_user}': {question}")

    raw_product_string = state.product_item
    print(f"Raw extracted product string: '{raw_product_string}'")

    product_items = (
        [item.strip() for item in raw_product_string.split(",")]
        if raw_product_string
        else []
    )
    print(f"Extracted product items: {product_items}")

    session = SessionLocal()
    try:
        products = session.query(Product).all()
        product_names = [product.name for product in products]
        session.close()
    except Exception as e:
        session.close()
        raise e

    corrected_product_items = []
    original_to_corrected = {}

    if len(product_items) == 0:
        product_exists = False
        print(f"No product items found, setting product_exists to {product_exists}")
        return AgentState(**state.model_dump() | {"product_exists": product_exists})

    system = """You are an assistant that checks if product items exist in the stock, handling typos or variations.
The user asked: '{question}'.
The extracted product items are: {product_items}.
Here’s the list of available product names from the database: {product_names}.
- For each product item in {product_items}, return the EXACT matching product name from {product_names} (case-insensitive, allowing typos or variations).
- Correct common misspellings (e.g., 'margaritta' should match 'Margherita', 'spaghetty' should match 'Spaghetti', 'lazania' should match 'Lasagna').
- If no match is found for an item, return 'NOT_FOUND' for that item.
- Output a Python list of strings, one per product item, in the same order as {product_items}, with NO explanations or extra text.
Example:
- product_items: ["solar interaction lamp", "Generic Boîte Lunch", "Presse Argume Silver Crest"]
- product_names: ["Solar Interaction Wall Lamp", "Generic Boîte Lunch Box", "Presse Agrume Silver Crest"]
- Output: ["Solar Interaction Wall Lamp", "Generic Boîte Lunch Box", "Presse Agrume Silver Crest"]
- product_items: ["pizza margaritta", "spaghetty carbonara", "lazania"]
- product_names: ["Pizza Margherita", "Spaghetti Carbonara", "Lasagna"]
- Output: ["Pizza Margherita", "Spaghetti Carbonara", "Lasagna"]
"""

    check_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system.format(
                    question=question,
                    product_items=product_items,
                    product_names=product_names,
                ),
            ),
            ("human", "Does this product exist?"),
        ]
    )
    llm = ChatOllama(model="mistral", temperature=0, stream=False)
    chain = check_prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "question": question,
            "product_items": product_items,
            "product_names": product_names,
        }
    ).strip()
    print(f"LLM product check result for {product_items}: '{result}'")

    try:
        # Parse the LLM result as a Python list
        corrected_list = ast.literal_eval(result)
        if not isinstance(corrected_list, list) or len(corrected_list) != len(
            product_items
        ):
            raise ValueError("Result is not a valid list matching product_items length")

        product_exists = True
        for orig, corrected in zip(product_items, corrected_list):
            if corrected == "NOT_FOUND":
                product_exists = False
                corrected_product_items.append(orig)  # Keep original if not found
            else:
                corrected_product_items.append(corrected)
                original_to_corrected[orig] = corrected

    except (ValueError, SyntaxError) as e:
        print(f"Failed to parse LLM result: {e}. Treating as NOT_FOUND.")
        product_exists = False
        corrected_product_items = product_items[:]

    # Correct the question using regex
    if corrected_product_items and original_to_corrected:
        corrected_question = question
        for orig, corrected in original_to_corrected.items():
            pattern = rf"\b{re.escape(orig)}\b"
            corrected_question = re.sub(
                pattern, corrected, corrected_question, flags=re.IGNORECASE
            )
        state.question = corrected_question
        print(f"Question product item corrected: '{state.question}'")

    print(
        f"State updated - Product exists: {product_exists}, Corrected product items: {corrected_product_items}"
    )
    return AgentState(
        **state.model_dump()
        | {
            "product_exists": product_exists,
            "corrected_product_items": corrected_product_items,
        }
    )
