# Directory: llm_chatbot_server/chatbot/
# Purpose: Defines handler functions for processing user inputs and generating responses.
# Prerequisites:
#   - Dependencies: langchain, langchain_core, sqlalchemy.
#   - chatbot/llm.py for llm, extract_answer.
#   - chatbot/api.py for api_call.
# Notes:
#   - process_input relies solely on LLM for item extraction, ensuring requested_items is a list.
#   - handle_new_order validates input types and matches all requested items in a single LLM call.

import logging
from langchain_core.messages import HumanMessage
from chatbot.llm import llm, extract_answer
from chatbot.api import api_call

logger = logging.getLogger(__name__)


def process_input(state: dict) -> dict:
    """
    Classify the user input into an intent and extract relevant information.
    Ensures requested_items is a list of strings using only LLM extraction.
    """
    user_input = state.get("user_input", "").strip()
    language = state.get("language", "english").lower()
    next_step = state.get("next_step", None)

    if next_step == "await_address":
        logger.info(f"Skipping intent detection for address input: {user_input}")
        return {
            **state,
            "user_input": user_input,
            "intent": "none",
            "address": user_input,
        }

    if not user_input or user_input == "":
        logger.warning("Empty user input, setting intent to none")
        return {
            **state,
            "intent": "none",
            "requested_items": [],
            "issue_product": "none",
            "address": state.get("address", "none"),
        }

    # Define intent classification prompt by language with E-commerce context
    prompt = (
        f"You are an E-commerce Agent assisting customers with requests about products and orders. "
        f"Classify the intent of the user's input: '{user_input}'. "
        f"Possible intents: new_order, retrieve_order, list_products, greeting, report_issue, none. "
        f"If the user wants to purchase items (e.g., 'I want to buy X and Y' or 'I want to order the Z'), classify as 'new_order' and extract the item names exactly as provided (e.g., 'Presse Argume' or 'Presse Argume,wall lamp'). "
        f"For a single item, output only that item (e.g., 'Presse Argume'). For multiple items, separate by commas without 'and' (e.g., 'X,Y'). "
        f"If no items are specified or the input is ambiguous, set Items to 'none'. "
        f"Do not use 'Non-relevant' or other invalid values. "
        f"Output exactly in this format:\n"
        f"**Intent:** intent_name\n"
        f"**Items:** item_name_1,item_name_2,...,item_name_n_or_none\n"
        f"**IssueProduct:** none\n"
        f"**Address:** none"
    )

    message = HumanMessage(content=prompt)

    try:
        logger.info(
            f"Streaming LLM response for intent classification of '{user_input}':"
        )
        response = ""
        for chunk in llm.stream([message]):
            chunk_content = chunk.content
            response += chunk_content
            print(chunk_content, end="", flush=True)
        print()
        # response = llm.invoke([message]).content
        # logger.info(f"Complete LLM procees_input response for intent: {response}")

        intent = extract_answer(response, "**Intent:**")
        requested_items_raw = extract_answer(response, "**Items:**")
        issue_product = extract_answer(response, "**IssueProduct:**")
        address = extract_answer(response, "**Address:**")

        valid_intents = {
            "new_order",
            "retrieve_order",
            "list_products",
            "greeting",
            "report_issue",
            "none",
        }
        if intent not in valid_intents:
            logger.warning(f"Invalid intent detected: {intent}, defaulting to 'none'")
            intent = "none"
            requested_items_raw = "none"

        # Ensure requested_items is a list using LLM output only
        if intent == "new_order":
            if (
                requested_items_raw == "none"
                or not requested_items_raw
                or requested_items_raw.lower() in ["non-relevant", ""]
            ):
                logger.warning(
                    f"No valid items extracted from LLM output: {requested_items_raw}"
                )
                requested_items = []
            else:
                # Convert LLM output to list
                requested_items = (
                    [
                        item.strip()
                        for item in requested_items_raw.split(",")
                        if item.strip()
                    ]
                    if "," in requested_items_raw
                    else [requested_items_raw.strip()]
                )
                logger.info(
                    f"Using LLM-extracted items: {requested_items}, type: {type(requested_items)}"
                )
        else:
            requested_items = []

        if intent == "list_products":
            requested_items = []

        state = {
            **state,
            "intent": intent,
            "requested_items": requested_items,
            "issue_product": issue_product,
            "address": address,
        }
        logger.info(f"State after process_input: {state}")
        return state

    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        state = {
            **state,
            "intent": "none",
            "requested_items": [],
            "issue_product": "none",
            "address": state.get("address", "none"),
        }
        logger.info(f"State after process_input (error): {state}")
        return state


def handle_list_products(state: dict) -> dict:
    """
    Generate a response listing available products in the user's language.
    """
    language = state.get("language", "english")
    user_input = state.get("user_input", "")

    logger.info(
        f"Handling list_products intent for input '{user_input}' in language '{language}'"
    )

    try:
        products = api_call("list_products")
        if "error" in products:
            logger.error(f"Failed to fetch products: {products['error']}")
            products = []
        else:
            products = [{"name": p["name"], "price": p["price"]} for p in products]
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        products = []

    if not products:
        logger.warning("No products available to list")
        if language == "french":
            state["response"] = "Désolé, aucun produit n'est disponible pour le moment."
        elif language == "arabic":
            state["response"] = "عذرًا، لا توجد منتجات متاحة في الوقت الحالي."
        else:
            state["response"] = "Sorry, no products are available at the moment."
        return state

    # Prepare product list in the correct language
    product_list = ", ".join(
        [
            f"{p['name']} ({p['price']} {'د.ت' if language == 'arabic' else 'TND' if language == 'french' else '$'})"
            for p in products
        ]
    )
    if language == "french":
        response = f"Voici nos produits : {product_list}. Comment puis-je vous aider ?"
    elif language == "arabic":
        response = f"هذه منتجاتنا: {product_list}. كيف يمكنني مساعدتك؟"
    else:
        response = f"Here are our products: {product_list}. How can I assist you?"

    try:
        prompt = (
            f"You are an E-commerce Agent assisting customers. "
            f"Generate a friendly message listing available products based on the input '{user_input}'. "
            f"Available products: {', '.join([p['name'] + ' (' + str(p['price']) + ' TND)' for p in products])}. "
            f"List only product names and prices, no identifiers. "
            f"Do not translate or modify product names; use them exactly as provided. "
            f"Keep it short, natural, and professional in {language}. "
            f"Example in French: 'Voici nos produits : Presse Agrume Silver Crest YZI-001 45W Rose (38 TND), solar interaction wall lamp (23 TND). Comment puis-je vous aider ?' "
            f"Output exactly in this format:\n"
            f"**Response:** message_liste_produits"
        )
        message = HumanMessage(content=prompt)
        llm_response = llm.invoke([message]).content
        logger.info(f"Raw LLM response: {llm_response}")

        generated_response = extract_answer(llm_response, "**Response:**")
        if generated_response:
            response = generated_response

    except Exception as e:
        logger.error(f"Error generating product list: {str(e)}")

    state["response"] = response
    logger.info(f"Generated product list in {language}: {response}")
    return state


def handle_none(state: dict) -> dict:
    """
    Handle unclear or unrecognized intents by asking for clarification.
    """
    language = state.get("language", "english")
    user_input = state.get("user_input", "")

    if language == "french":
        response = "Désolé, je n’ai pas compris votre demande. Pouvez-vous préciser, comme lister nos produits ou vérifier une commande ?"
    elif language == "arabic":
        response = (
            "عذرًا، لم أفهم طلبك. هل يمكنك التوضيح، مثل سرد المنتجات أو التحقق من طلب؟"
        )
    else:
        response = "Sorry, I didn’t understand your request. Could you clarify, like listing our products or checking an order?"

    try:
        prompt = (
            f"You are an E-commerce Agent assisting customers. "
            f"The user's input '{user_input}' was unclear. "
            f"Generate a friendly, professional clarification message in {language}. "
            f"Suggest options like listing products or checking an order. "
            f"Keep it short and natural. "
            f"Example in English: 'Sorry, I didn’t understand your request. Could you clarify, like listing our products or checking an order?' "
            f"Output exactly in this format:\n"
            f"**Response:** clarification_message"
        )
        message = HumanMessage(content=prompt)
        logger.info(f"Streaming LLM response for clarification in {language}:")
        llm_response = ""
        for chunk in llm.stream([message]):
            chunk_content = chunk.content
            llm_response += chunk_content
            print(chunk_content, end="", flush=True)

        # llm_response = llm.invoke([message]).content
        # logger.info(f"Raw LLM response: {llm_response}")

        generated_response = extract_answer(llm_response, "**Response:**")
        if generated_response:
            response = generated_response

    except Exception as e:
        logger.error(f"Error generating clarification: {str(e)}")

    state["response"] = response
    logger.info(f"Generated clarification in {language}: {response}")
    return state


def handle_greeting(state: dict) -> dict:
    """
    Handle greeting intents with a friendly response.
    """
    logger.info(state)
    language = state.get("language", "english")
    logger.info(f"Handling greeting in {language}")
    user_input = state.get("user_input", "")

    if language == "french":
        response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
    elif language == "arabic":
        response = "مرحبًا! كيف يمكنني مساعدتك اليوم؟"
    else:
        response = "Hello! How can I assist you today?"

    try:
        prompt = (
            f"You are an E-commerce Agent assisting customers. "
            f"The user provided a greeting: '{user_input}'. "
            f"Generate a friendly, professional greeting response in {language}. "
            f"Keep it short and natural. "
            f"Example in English: 'Hello! How can I assist you today?' "
            f"Output exactly in this format:\n"
            f"**Response:** greeting_message"
        )
        message = HumanMessage(content=prompt)
        llm_response = llm.invoke([message]).content
        logger.info(f"Raw LLM response: {llm_response}")

        generated_response = extract_answer(llm_response, "**Response:**")
        if generated_response:
            response = generated_response

    except Exception as e:
        logger.error(f"Error generating greeting: {str(e)}")

    state["response"] = response
    logger.info(f"Generated greeting in {language}: {response}")
    return state


def handle_new_order(state: dict, config: dict) -> dict:
    """
    Handle new_order intents by matching requested products and creating an order.
    Matches all requested items in a single LLM call, no fuzzy matching.
    """
    user_id = config.get("configurable", {}).get("phone_number", "unknown")
    print(f"User ID: {user_id}")
    # Strip 'whatsapp:+216' to get last 8 digits
    if user_id.startswith("whatsapp:+216"):
        user_id_converty = user_id.replace("whatsapp:+216", "")
    else:
        user_id_converty = user_id[-8:] if len(user_id) >= 8 else user_id
        logger.warning(
            f"Unexpected phone number format: {user_id}, using {user_id_converty}"
        )
    print(f"User ID (converted): {user_id_converty}")
    language = state.get("language", "english")
    requested_items = state.get("requested_items", [])
    user_input = state.get("user_input", "")

    # Validate requested_items as a list
    if isinstance(requested_items, str):
        logger.warning(
            f"requested_items is a string: {requested_items}, converting to list"
        )
        requested_items = [requested_items.strip()]
    elif not isinstance(requested_items, list):
        logger.error(
            f"Invalid requested_items type: {type(requested_items)}, expected list"
        )
        if language == "french":
            response = "Désolé, une erreur s’est produite avec votre commande. Pouvez-vous préciser les produits ?"
        elif language == "arabic":
            response = "عذرًا، حدث خطأ في طلبك. هل يمكنك تحديد المنتجات؟"
        else:
            response = "Sorry, an error occurred with your order. Could you specify the products?"
        state["response"] = response
        return state
    elif not requested_items:
        logger.error(f"No requested items provided for new_order: {user_input}")
        if language == "french":
            response = "Désolé, aucun produit n’a été spécifié. Pouvez-vous préciser le nom du produit ?"
        elif language == "arabic":
            response = "عذرًا، لم يتم تحديد أي منتج. هل يمكنك تحديد اسم المنتج؟"
        else:
            response = (
                "Sorry, no products were specified. Could you specify the product name?"
            )
        state["response"] = response
        return state

    # Fetch user data
    user_data = api_call(
        "get_user",
        {
            "user_id": user_id,
            "name": config.get("configurable", {}).get("name", "Unknown"),
        },
    )
    existing_address = user_data.get("address", "none")

    logger.info(
        f"Handling new_order intent for input '{user_input}' with requested_items: {requested_items}, user_id: {user_id_converty}"
    )

    # Fetch products
    try:
        products = api_call("list_products")
        if "error" in products:
            logger.error(f"Failed to fetch products: {products['error']}")
            products = []
        else:
            products = [{"name": p["name"], "price": p["price"]} for p in products]
        logger.info(f"Fetched products: {products}")
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        products = []
    print(f"Products: {products}")

    # Attempt LLM-based product matching for all requested items
    matched_products = []
    if requested_items and products:
        print(type(requested_items), "in prompt")
        prompt = (
            f"You are an E-commerce Agent assisting customers. "
            f"The requested items are: {', '.join(requested_items)}. "
            f"There are exactly {len(requested_items)} item(s) to match. "
            f"Available products: {', '.join([p['name'] for p in products])}. "
            f"For each requested item, identify the most likely matching product from the available products. "
            f"Account for misspellings, shortened names, or partial matches by prioritizing string similarity. "
            f"For example, 'Presse Argume' should match 'Presse Agrume Silver Crest YZI-001 45W Rose' because 'Argume' and 'Agrume' differ by only one letter, and 'Presse' is an exact match. "
            f"Use the following rules for matching: "
            f"1. Prioritize products where the requested item is a substring of the product name (ignoring case). "
            f"2. If no substring match, select the product with the closest string similarity (e.g., smallest number of letter changes). "
            f"3. If no reasonable match is found, use 'none'. "
            f"Return exactly {len(requested_items)} product name(s) in a comma-separated string, one for each requested item. "
            f"Do not return extra products, duplicate matches, or items not in the requested list. "
            f"Ignore any other items mentioned in the user input or elsewhere. "
            f"Preserve spaces in product names and do not use underscores or other separators. "
            f"Output exactly in this format:\n"
            f"**Products:** product_name_1,product_name_2,...,product_name_n"
        )
        message = HumanMessage(content=prompt)
        try:
            print(f"LLM call for product matching: {', '.join(requested_items)}")

            response = ""
            for chunk in llm.stream([message]):
                chunk_content = chunk.content
                response += chunk_content
                print(chunk_content, end="", flush=True)
            print()

            # response = llm.invoke([message]).content
            # print(f"LLM response: {response}")
            matched_products = extract_answer(response, "**Products:**")
            # Truncate to match requested_items length
            matched_products = matched_products[: len(requested_items)]
            if not matched_products or len(matched_products) < len(requested_items):
                logger.warning(
                    f"LLM returned insufficient matches for {requested_items}: {matched_products}"
                )
                matched_products = matched_products + ["none"] * (
                    len(requested_items) - len(matched_products)
                )
            print(f"Matched products (LLM): {matched_products}")
        except Exception as e:
            logger.error(f"Error in LLM product matching: {str(e)}")
            print(f"LLM error: {str(e)}")
            matched_products = ["none"] * len(requested_items)

    # If no valid products matched
    if not matched_products or all(p == "none" for p in matched_products):
        if language == "french":
            response = "Désolé, je n’ai pas trouvé les produits que vous souhaitez commander. Pouvez-vous préciser les noms des produits ?"
        elif language == "arabic":
            response = (
                "عذرًا، لم أجد المنتجات التي تريد طلبها. هل يمكنك تحديد أسماء المنتجات؟"
            )
        else:
            response = "Sorry, I couldn’t find the products you want to order. Could you specify the product names?"
        state["response"] = response
        logger.info(f"Generated response for invalid products: {response}")
        return state

    # Verify all matched products exist
    valid_products = []
    for product in matched_products:
        if product != "none" and any(
            p["name"].lower() == product.lower() for p in products
        ):
            valid_products.append(product)
        else:
            logger.warning(f"Product not found in list or marked none: {product}")

    if not valid_products:
        if language == "french":
            response = f"Les produits {', '.join(matched_products)} ne sont pas disponibles. Voulez-vous voir nos produits ?"
        elif language == "arabic":
            response = f"المنتجات {', '.join(matched_products)} غير متوفرة. هل تريد رؤية منتجاتنا؟"
        else:
            response = f"The products {', '.join(matched_products)} are not available. Would you like to see our products?"
        state["response"] = response
        logger.info(f"Generated response for unavailable products: {response}")
        return state

    # Update state with valid products
    state["requested_items"] = valid_products
    print(f"Updated requested_items: {state['requested_items']}")
    print(f"Address:", state["address"])

    # Handle address
    if existing_address != "none" and existing_address != "unknown":
        print(f"Existing address: {existing_address}")
        try:
            order_data = {
                "items": valid_products,
                "status": "Pending",
            }
            result = api_call(
                "new_order",
                {
                    "user_id": user_id_converty,
                    "address": existing_address,
                    "order_data": order_data,
                },
            )
            if "error" in result:
                logger.error(f"Failed to create order: {result['error']}")
                if language == "french":
                    response = f"Une erreur s’est produite lors de la création de votre commande pour {', '.join(valid_products)}. Veuillez réessayer."
                elif language == "arabic":
                    response = f"حدث خطأ أثناء إنشاء طلبك لـ {', '.join(valid_products)}. يرجى المحاولة مرة أخرى."
                else:
                    response = f"An error occurred while creating your order for {', '.join(valid_products)}. Please try again."
                state["response"] = response
            else:
                state["order_data"] = result
                prompt = (
                    f"Generate a confirmation message in {language} for an order of {', '.join(valid_products)} with Order ID {result['order_id']}. "
                    f"Include the delivery address '{existing_address}'. "
                    f"Keep it short and friendly. "
                    f"Example in English: 'Your order for {', '.join(valid_products)} has been confirmed (ID: {result['order_id']}). We’ll deliver to {existing_address}. Thank you!' "
                    f"Output exactly in this format:\n"
                    f"**Response:** confirmation_message"
                )
                message = HumanMessage(content=prompt)
                response = llm.invoke([message]).content
                state["response"] = extract_answer(response, "**Response:**")
                state["next_step"] = None
                state["requested_items"] = []
        except Exception as e:
            logger.error(f"Error in handle_new_order with existing address: {str(e)}")
            if language == "french":
                response = f"Une erreur s’est produite lors de la création de votre commande pour {', '.join(valid_products)}. Veuillez réessayer."
            elif language == "arabic":
                response = f"حدث خطأ أثناء إنشاء طلبك لـ {', '.join(valid_products)}. يرجى المحاولة مرة أخرى."
            else:
                response = f"An error occurred while creating your order for {', '.join(valid_products)}. Please try again."
            state["response"] = response
    else:
        prompt = (
            f"Generate a message in {language} asking for the user's address to order {', '.join(valid_products)}. "
            f"Keep it short and friendly. "
            f"Example in English: 'Please provide your address to order {', '.join(valid_products)}.' "
            f"Output exactly in this format:\n"
            f"**Response:** address_request_message"
        )
        message = HumanMessage(content=prompt)
        try:
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
            state["next_step"] = "await_address"
        except Exception as e:
            logger.error(f"Error generating address request: {str(e)}")
            if language == "french":
                response = f"Veuillez fournir votre adresse pour commander {', '.join(valid_products)}."
            elif language == "arabic":
                response = f"يرجى تقديم عنوانك لطلب {', '.join(valid_products)}."
            else:
                response = (
                    f"Please provide your address to order {', '.join(valid_products)}."
                )
            state["response"] = response
            state["next_step"] = "await_address"

    logger.info(f"State after handle_new_order: {state}")
    return state


def handle_address_input(state: dict, config: dict) -> dict:
    """
    Handle user input as an address for a new order, update address, and create the order.
    Supports multiple products in requested_items.
    """
    user_id = config.get("configurable", {}).get("phone_number", "unknown")
    if user_id.startswith("whatsapp:+216"):
        user_id_converty = user_id.replace("whatsapp:+216", "")
    else:
        user_id_converty = user_id[-8:] if len(user_id) >= 8 else user_id
        logger.warning(
            f"Unexpected phone number format: {user_id}, using {user_id_converty}"
        )
    language = state.get("language", "english")
    user_input = state.get("user_input", "").strip()
    requested_items = state.get("requested_items", [])

    logger.info(
        f"Handling address input '{user_input}' for user_id: {user_id_converty}, requested_items: {requested_items}"
    )

    if not user_input:
        if language == "french":
            response = f"Veuillez fournir une adresse valide pour commander {', '.join(requested_items)}."
        elif language == "arabic":
            response = f"يرجى تقديم عنوان صالح لطلب {', '.join(requested_items)}."
        else:
            response = (
                f"Please provide a valid address to order {', '.join(requested_items)}."
            )
        state["response"] = response
        state["next_step"] = "await_address"
        logger.info(f"Generated response for empty address: {response}")
        return state

    if not requested_items:
        if language == "french":
            response = "Erreur : aucun produit sélectionné. Veuillez indiquer les produits que vous souhaitez commander."
        elif language == "arabic":
            response = (
                "خطأ: لم يتم اختيار أي منتج. يرجى تحديد المنتجات التي تريد طلبها."
            )
        else:
            response = "Error: No products selected. Please specify the products you want to order."
        state["response"] = response
        state["next_step"] = None
        logger.info(f"Generated response for missing products: {response}")
        return state

    try:
        # Update address
        api_call("update_address", {"user_id": user_id, "address": user_input})
        # Create order
        order_data = {"items": requested_items, "status": "Pending"}
        result = api_call(
            "new_order", {"user_id": user_id_converty, "order_data": order_data}
        )
        if "error" in result:
            logger.error(f"Failed to create order: {result['error']}")
            if language == "french":
                response = f"Une erreur s’est produite lors de la création de votre commande pour {', '.join(requested_items)}. Veuillez réessayer."
            elif language == "arabic":
                response = f"حدث خطأ أثناء إنشاء طلبك لـ {', '.join(requested_items)}. يرجى المحاولة مرة أخرى."
            else:
                response = f"An error occurred while creating your order for {', '.join(requested_items)}. Please try again."
        else:
            state["order_data"] = result
            prompt = (
                f"Generate a confirmation message in {language} for an order of {', '.join(requested_items)} with Order ID {result['order_id']}. "
                f"Include the delivery address '{user_input}'. "
                f"Keep it short and friendly. "
                f"Example in English: 'Your order for {', '.join(requested_items)} has been confirmed (ID: {result['order_id']}). We’ll deliver to {user_input}. Thank you!' "
                f"Output exactly in this format:\n"
                f"**Response:** confirmation_message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
            state["next_step"] = None
            state["requested_items"] = []
            state["address"] = None
    except Exception as e:
        logger.error(f"Error in handle_address_input: {str(e)}")
        if language == "french":
            response = f"Une erreur s’est produite lors de la création de votre commande pour {', '.join(requested_items)}. Veuillez réessayer."
        elif language == "arabic":
            response = f"حدث خطأ أثناء إنشاء طلبك لـ {', '.join(requested_items)}. يرجى المحاولة مرة أخرى."
        else:
            response = f"An error occurred while creating your order for {', '.join(requested_items)}. Please try again."
        state["response"] = response

    logger.info(f"State after handle_address_input: {state}")
    return state


def retrieve_order(state: dict, config: dict) -> dict:
    print(f"Retrieving order for state: {state}")
    user_id = config.get("configurable", {}).get("phone_number")
    language = state["language"]
    try:
        orders = api_call("get_orders", {"user_id": user_id})
        state["order_data"] = orders
        if orders:
            order_lines = []
            for order in orders:
                items_str = (
                    " and ".join(order["items"])
                    if isinstance(order["items"], list)
                    else order["items"]
                )
                order_lines.append(
                    f"- Order ID: {order['order_id']}, Items: {items_str}, Status: {order['status']}"
                )
            prompt = (
                f"Generate a message in {language} listing the user's orders: {', '.join(order_lines)}. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
        else:
            prompt = (
                f"Generate a message in {language} informing the user that no orders were found and suggesting they start shopping. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
    except Exception as e:
        logger.error(f"Error in retrieve_order: {e}")
        prompt = (
            f"Generate an error message in {language} indicating a failure to fetch orders: {str(e)}. "
            f"Keep it short and friendly. "
            f"Output exactly in this format:\n"
            f"**Response:** error_message"
        )
        message = HumanMessage(content=prompt)
        response = llm.invoke([message]).content
        state["response"] = extract_answer(response, "**Response:**")
    return state


def handle_report_issue(state: dict, config: dict) -> dict:
    user_id = config.get("configurable", {}).get("phone_number")
    language = state["language"]
    name = config.get("configurable", {}).get("name")
    issue_product = state.get("issue_product")
    user_input = state["user_input"]

    if not issue_product or issue_product == "none":
        prompt = (
            f"Generate a message in {language} informing the user that no product was identified and asking them to specify a product they’ve ordered (e.g., 'problem with my phone'). "
            f"Keep it short and friendly. "
            f"Output exactly in this format:\n"
            f"**Response:** message"
        )
        message = HumanMessage(content=prompt)
        response = llm.invoke([message]).content
        state["response"] = extract_answer(response, "**Response:**")
    else:
        # Retrieve user's orders
        orders = api_call("get_orders", {"user_id": user_id})
        ordered_items = []
        for order in orders:
            items = order.get("items", [])
            if isinstance(items, list):
                ordered_items.extend(items)
            else:
                ordered_items.append(items)

        if not ordered_items:
            # No orders found, inform user
            prompt = (
                f"Generate a message in {language} informing the user that no orders were found and suggesting they start shopping or check their order history. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
        else:
            # Use LLM to match issue_product to ordered items
            prompt = (
                f"You are an E-commerce Agent assisting customers. "
                f"The user reported an issue with: '{', '.join(issue_product)}'. "
                f"Their ordered items are: {', '.join(ordered_items)}. "
                f"There is exactly 1 item to match. "
                f"Identify the most likely matching product from the ordered items, accounting for misspellings, shortened names, or partial matches. "
                f"For example, 'Presse Argume' should match 'Presse Agrume Silver Crest YZI-001 45W Rose' because 'Argume' and 'Agrume' differ by one letter, and 'Presse' is an exact match. "
                f"Use the following rules: "
                f"1. Prioritize products where the issue item is a substring of the ordered item name (ignoring case). "
                f"2. If no substring match, select the item with the closest string similarity (e.g., smallest number of letter changes). "
                f"3. If no reasonable match is found, return 'none'. "
                f"Return exactly 1 product name in a comma-separated string. "
                f"Preserve spaces in product names and do not use underscores. "
                f"Output exactly in this format:\n"
                f"**Products:** product_name"
            )
            message = HumanMessage(content=prompt)
            try:
                response = llm.invoke([message]).content
                logger.info(f"LLM response for issue product matching: {response}")
                matched_product = extract_answer(response, "**Products:**")

                # Handle cases where extract_answer returns a list
                if isinstance(matched_product, list):
                    matched_product = matched_product[0] if matched_product else "none"

                logger.info(f"Matched product: {matched_product}")

                if matched_product.lower() == "none" or not any(
                    matched_product.lower() == item.lower() for item in ordered_items
                ):
                    # No match found, list user's orders to help them choose
                    order_lines = []
                    for order in orders:
                        items_str = (
                            " and ".join(order["items"])
                            if isinstance(order["items"], list)
                            else order["items"]
                        )
                        order_lines.append(
                            f"- Order ID: {order['order_id']}, Items: {items_str}, Status: {order['status']}"
                        )
                    prompt = (
                        f"Generate a message in {language} informing the user that '{issue_product}' was not found in their orders. "
                        f"List their orders: {', '.join(order_lines)}. "
                        f"Ask them to specify a purchased product (e.g., 'problem with my phone'). "
                        f"Keep it short and friendly. "
                        f"Output exactly in this format:\n"
                        f"**Response:** message"
                    )
                    message = HumanMessage(content=prompt)
                    response = llm.invoke([message]).content
                    state["response"] = extract_answer(response, "**Response:**")
                else:
                    # Use LLM to categorize the claim
                    claim_categories = [
                        "defective",
                        "wrong_item",
                        "missing_item",
                        "delivery",
                        "quality",
                        "quantity",
                        "packaging",
                        "other",
                    ]
                    prompt = (
                        f"You are an E-commerce Agent categorizing a customer claim. "
                        f"The user reported an issue with '{matched_product}' and described it as: '{user_input}'. "
                        f"Categorize the issue into one of the following categories: {', '.join(claim_categories)}. "
                        f"- 'defective': Product is damaged or not functioning (e.g., 'doesn’t work', 'broken'). "
                        f"- 'wrong_item': Received a different product (e.g., 'got a lamp instead'). "
                        f"- 'missing_item': Product or parts missing (e.g., 'missing blades'). "
                        f"- 'delivery': Shipping issues like late or non-delivery (e.g., 'hasn’t arrived'). "
                        f"- 'quality': Poor quality or below expectations (e.g., 'feels cheap'). "
                        f"- 'quantity': Incorrect number of items (e.g., 'got two instead of one'). "
                        f"- 'packaging': Damaged due to poor packaging (e.g., 'box was crushed'). "
                        f"- 'other': Issues not fitting above (e.g., 'just unhappy'). "
                        f"Analyze the description and select the most appropriate category. "
                        f"If unclear, default to 'other'. "
                        f"Output exactly in this format:\n"
                        f"**Category:** category_name"
                    )
                    message = HumanMessage(content=prompt)
                    response = llm.invoke([message]).content
                    logger.info(f"LLM response for claim categorization: {response}")
                    claim_category = extract_answer(response, "**Category:**")

                    # Validate category
                    if isinstance(claim_category, list):
                        claim_category = (
                            claim_category[0] if claim_category else "other"
                        )
                    if claim_category not in claim_categories:
                        logger.warning(
                            f"Invalid category '{claim_category}', defaulting to 'other'"
                        )
                        claim_category = "other"

                    logger.info(f"Claim category: {claim_category}")

                    # Save the issue with category
                    issue_data = {
                        "product": matched_product,
                        "description": user_input,
                        "name": name,
                        "phone_number": user_id,
                        "status": "Pending",
                        "type": claim_category,
                    }
                    result = api_call(
                        "save_issue", {"user_id": user_id, "issue": issue_data}
                    )
                    prompt = (
                        f"You are an E-commerce Agent generating a response in {language} ONLY. Do not use any other language, including Chinese. "
                        f"Generate a message in {language} thanking the user for reporting an issue with {matched_product} and informing them an agent will contact them soon. "
                        f"Include Issue ID: {result['issue_id']}. "
                        f"Mention the issue category ({claim_category}) for clarity. "
                        f"Keep it short and friendly. "
                        f"Output exactly in this format:\n"
                        f"**Response:** message"
                    )
                    message = HumanMessage(content=prompt)
                    response = llm.invoke([message]).content
                    state["response"] = extract_answer(response, "**Response:**")
            except Exception as e:
                logger.error(
                    f"Error in LLM processing for issue product or category: {str(e)}"
                )
                prompt = (
                    f"Generate an error message in {language} indicating a failure to process the issue for '{issue_product}': {str(e)}. "
                    f"Ask the user to try again or contact support. "
                    f"Keep it short and friendly. "
                    f"Output exactly in this format:\n"
                    f"**Response:** message"
                )
                message = HumanMessage(content=prompt)
                response = llm.invoke([message]).content
                state["response"] = extract_answer(response, "**Response:**")

    state["next_step"] = None
    state["issue_product"] = None
    logger.info(f"State after handle_report_issue: {state}")
    return state


def generate_response(state: dict, config: dict) -> dict:
    language = state["language"]
    if (
        state.get("order_data")
        and "order_id" in state["order_data"]
        and state.get("intent") in ["new_order", "retrieve_order"]
    ):
        order_id = state["order_data"]["order_id"]
        prompt = (
            f"Generate a short, friendly confirmation message in {language} for Order ID: {order_id}. "
            f"Output exactly in this format:\n"
            f"**Response:** message"
        )
        try:
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            logger.info(f"LLM response for confirmation: {response}")
            state["response"] = extract_answer(response, "**Response:**")
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            prompt = (
                f"Generate an error message in {language} indicating a failure to confirm the order: {str(e)}. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** error_message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
    elif not state.get("response"):
        prompt = (
            f"Generate a short, friendly message in {language} asking what the user needs help with today. "
            f"Output exactly in this format:\n"
            f"**Response:** message"
        )
        message = HumanMessage(content=prompt)
        response = llm.invoke([message]).content
        state["response"] = extract_answer(response, "**Response:**")
    logger.info(f"State after generate_response: {state}")
    return state
