# Directory: llm_chatbot_server/chatbot/
# Purpose: Contains LangGraph node handlers for processing user intents and generating responses (e.g., greeting, new order, report issue).
# Prerequisites:
#   - Dependencies: langchain, langchain-ollama, langgraph.
#   - chatbot/llm.py for llm and extract_answer.
#   - chatbot/api.py for api_call.
#   - chatbot/types.py for AgentState.
# Notes:
#   - Handles intents like greeting ("bonjour"), new_order, and report_issue.
#   - Largest module (~400 lines); consider splitting further if it grows (e.g., order_handlers.py).

import logging
import traceback
from langchain_core.messages import HumanMessage
from chatbot.types import AgentState
from chatbot.llm import llm, extract_answer
from chatbot.api import api_call
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def process_input(state: dict) -> dict:
    """
    Classify the user input into an intent and extract relevant information.
    """
    user_input = state.get("user_input", "").strip()
    language = state.get("language", "english").lower()
    next_step = state.get("next_step", None)
    if next_step == "await_address":
        logger.info(f"Skipping intent detection for address input: {user_input}")
        return {
            **state,
            "user_input": user_input,
            "intent": "none",  # Set to none to avoid misrouting
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
    intent_templates = {
        "english": (
            f"You are an E-commerce Agent assisting customers with requests about products and orders. "
            f"Classify the intent of the user's input: '{user_input}'. "
            f"Possible intents: new_order, retrieve_order, list_products, greeting, report_issue, none. "
            f"If the user wants to purchase items (e.g., 'I want to buy X and Y'), classify as 'new_order' and extract the item names exactly as provided (e.g., 'Presse Argume,wall lamp'). "
            f"If no items are specified or the input is ambiguous, set Items to 'none'. "
            f"Output exactly in this format:\n"
            f"**Intent:** intent_name\n"
            f"**Items:** item_name_1,item_name_2,...,item_name_n\n"
            f"**IssueProduct:** none\n"
            f"**Address:** none"
        ),
        "french": (
            f"Vous êtes un Agent E-commerce aidant les clients avec des demandes sur les produits, les commandes et les problèmes. "
            f"Classifier l'intention de l'entrée : '{user_input}'. "
            f"Intentions possibles : 'new_order', 'retrieve_order', 'list_products', 'greeting', 'report_issue'. "
            f"- 'new_order' : Demande d'achat d'articles (par exemple, 'Je veux acheter une lampe'). "
            f"- 'retrieve_order' : Demande de vérification des commandes existantes (par exemple, 'Montrez mes commandes'). "
            f"- 'list_products' : Demande de liste des produits disponibles (par exemple, 'Que vendez-vous ?', 'Quels produits avez-vous ?'). "
            f"- 'greeting' : Salutations simples (par exemple, 'bonsoir', 'bonjour'). "
            f"- 'report_issue' : Signalement d'un problème (par exemple, 'Ma commande est cassée'). "
            f"Si aucune intention ne correspond, utiliser 'none'. "
            f"Sortie exactement dans ce format :\n"
            f"**Intent:** nom_intention\n**Items:** aucun\n**IssueProduct:** aucun\n**Address:** aucun"
        ),
        "arabic": (
            f"أنت وكيل تجارة إلكترونية تساعد العملاء في استفسارات المنتجات، الطلبات، والمشكلات. "
            f"تصنيف نية الإدخال: '{user_input}'. "
            f"النيات الممكنة: 'new_order', 'retrieve_order', 'list_products', 'greeting', 'report_issue'. "
            f"- 'new_order': طلب شراء عناصر (مثل 'أريد شراء مصباح'). "
            f"- 'retrieve_order': طلب التحقق من الطلبات الموجودة (مثل 'أرني طلباتي'). "
            f"- 'list_products': طلب قائمة المنتجات المتاحة (مثل 'ماذا تبيع؟', 'ما هي المنتجات المتوفرة؟'). "
            f"- 'greeting': تحيات بسيطة (مثل 'مرحبا'). "
            f"- 'report_issue': الإبلاغ عن مشكلة (مثل 'طلبي مكسور'). "
            f"إذا لم تتطابق أي نية، استخدم 'none'. "
            f"الإخراج بالضبط بهذا الشكل:\n"
            f"**Intent:** اسم_النية\n**Items:** لا شيء\n**IssueProduct:** لا شيء\n**Address:** لا شيء"
        ),
    }

    prompt = intent_templates.get(language, intent_templates["english"])
    message = HumanMessage(content=prompt)

    try:
        logger.info(
            f"Streaming LLM response for intent classification of '{user_input}':"
        )
        response = llm.invoke([message]).content
        logger.info(f"Complete LLM procees_input response for intent: {response}")

        intent = extract_answer(response, "**Intent:**")
        requested_items = extract_answer(response, "**Items:**")
        issue_product = extract_answer(response, "**IssueProduct:**").lower()
        address = extract_answer(response, "**Address:**").lower()

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
            requested_items = "none"

        if intent == "list_products":
            requested_items = "none"

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
        return {
            **state,
            "intent": "none",
            "requested_items": "none",
            "issue_product": "none",
            "address": "none",
        }


def handle_list_products(state: dict) -> dict:
    """
    Handle list_products intent and generate a response listing available products without IDs.
    Product names must be used exactly as provided by the API, without translation or modification.
    """
    intent = state.get("intent", "").lower()
    language = state.get("language", "english").lower()
    products = api_call(intent)
    user_input = state.get("user_input", "").strip()

    if not products:
        logger.warning("No products available to list")
        response_templates = {
            "english": "Sorry, no products are available at the moment.",
            "french": "Désolé, aucun produit n'est disponible pour le moment.",
            "arabic": "عذرًا، لا توجد منتجات متاحة في الوقت الحالي.",
        }
        return {
            **state,
            "response": response_templates.get(language, response_templates["english"]),
        }

    # Define response prompt by language with E-commerce context
    response_templates = {
        "english": (
            f"You are an E-commerce Agent assisting customers with product inquiries. "
            f"Generate a friendly message in English listing the available products based on the input '{user_input}'. "
            f"Available products: {', '.join([p['name'] + ' (' + str(p['price']) + ' TND)' for p in products])}. "
            f"List only product names and prices, without IDs. Do not translate or modify the product names; use them exactly as provided. "
            f"Keep it short, natural, and professional. "
            f"Example: 'Here are our products: Presse Agrume Silver Crest YZI-001 45W Rose (38 TND), solar interaction wall lamp (23 TND), Generic Boîte Lunch Box - Carré (18 TND). How can I assist you?' "
            f"Output exactly in this format:\n"
            f"**Response:** product_list_message"
        ),
        "french": (
            f"Vous êtes un Agent E-commerce aidant les clients avec des demandes sur les produits. "
            f"Générer un message amical en français listant les produits disponibles basé sur l'entrée '{user_input}'. "
            f"Produits disponibles : {', '.join([p['name'] + ' (' + str(p['price']) + ' TND)' for p in products])}. "
            f"Lister uniquement les noms et prix des produits, sans identifiants. Ne pas traduire ni modifier les noms des produits ; utilisez-les exactement tels qu'ils sont fournis. "
            f"Garder court, naturel et professionnel. "
            f"Exemple : 'Voici nos produits : Presse Agrume Silver Crest YZI-001 45W Rose (38 TND), solar interaction wall lamp (23 TND), Generic Boîte Lunch Box - Carré (18 TND). Comment puis-je vous aider ?' "
            f"Sortie exactement dans ce format :\n"
            f"**Response:** message_liste_produits"
        ),
        "arabic": (
            f"أنت وكيل تجارة إلكترونية تساعد العملاء في استفسارات المنتجات. "
            f"إنشاء رسالة ودية بالعربية تسرد المنتجات المتاحة بناءً على الإدخال '{user_input}'. "
            f"المنتجات المتاحة: {', '.join([p['name'] + ' (' + str(p['price']) + ' د.ت)' for p in products])}. "
            f"سرد أسماء المنتجات وأسعارها فقط، بدون معرفات. لا تترجم أو تعدل أسماء المنتجات؛ استخدمها كما هي مقدمة. "
            f"اجعلها قصيرة، طبيعية، واحترافية. "
            f"مثال: 'هذه منتجاتنا: Presse Agrume Silver Crest YZI-001 45W Rose (38 د.ت)، solar interaction wall lamp (23 د.ت)، Generic Boîte Lunch Box - Carré (18 د.ت). كيف يمكنني مساعدتك؟' "
            f"الإخراج بالضبط بهذا الشكل:\n"
            f"**Response:** رسالة_قائمة_المنتجات"
        ),
    }

    prompt = response_templates.get(language, response_templates["english"])
    message = HumanMessage(content=prompt)

    try:
        logger.info(f"Streaming LLM response for listing products in {language}:")
        response = llm.invoke([message]).content
        # logger.info(f"Raw LLM response: {response}")

        product_list_message = extract_answer(response, "**Response:**")
        if not product_list_message:
            logger.warning(
                f"No product list message extracted, using fallback for {language}"
            )
            fallback_messages = {
                "english": (
                    f"Here are our products: {', '.join([p['name'] + ' ($' + str(p['price']) + ')' for p in products])}. "
                    f"How can I assist you?"
                ),
                "french": (
                    f"Voici nos produits : {', '.join([p['name'] + ' (' + str(p['price']) + ' TND)' for p in products])}. "
                    f"Comment puis-je vous aider ?"
                ),
                "arabic": (
                    f"هذه منتجاتنا: {', '.join([p['name'] + ' (' + str(p['price']) + ' د.ت)' for p in products])}. "
                    f"كيف يمكنني مساعدتك؟"
                ),
            }
            product_list_message = fallback_messages.get(
                language, fallback_messages["english"]
            )

        logger.info(f"Generated product list in {language}: {product_list_message}")
        return {**state, "response": product_list_message}

    except Exception as e:
        logger.error(f"Error generating product list: {str(e)}")
        fallback_messages = {
            "english": (
                f"Here are our products: {', '.join([p['name'] + ' ($' + str(p['price']) + ')' for p in products])}. "
                f"How can I assist you?"
            ),
            "french": (
                f"Voici nos produits : {', '.join([p['name'] + ' (' + str(p['price']) + ' TND)' for p in products])}. "
                f"Comment puis-je vous aider ?"
            ),
            "arabic": (
                f"هذه منتجاتنا: {', '.join([p['name'] + ' (' + str(p['price']) + ' د.ت)' for p in products])}. "
                f"كيف يمكنني مساعدتك؟"
            ),
        }
        return {
            **state,
            "response": fallback_messages.get(language, fallback_messages["english"]),
        }


def handle_none(state: dict) -> dict:
    """
    Handle none intent by asking the user to clarify their request in the detected language.
    """
    language = state.get("language", "english").lower()
    user_input = state.get("user_input", "").strip()

    # Define clarification prompt by language with E-commerce context
    clarification_templates = {
        "english": (
            f"You are an E-commerce Agent assisting customers with product inquiries, orders, and issues. "
            f"The user input '{user_input}' could not be classified into a specific intent. "
            f"Generate a friendly message in English asking the user to clarify their request. "
            f"Suggest options like listing products or checking orders. "
            f"Keep it short, natural, and professional. "
            f"Example: 'Sorry, I didn’t understand your request. Could you clarify, like asking for our products or checking an order?' "
            f"Output exactly in this format:\n"
            f"**Response:** clarification_message"
        ),
        "french": (
            f"Vous êtes un Agent E-commerce aidant les clients avec des demandes sur les produits, les commandes et les problèmes. "
            f"L'entrée utilisateur '{user_input}' n'a pas pu être classée dans une intention spécifique. "
            f"Générer un message amical en français demandant à l'utilisateur de clarifier sa demande. "
            f"Suggérer des options comme lister les produits ou vérifier une commande. "
            f"Garder court, naturel et professionnel. "
            f"Exemple : 'Désolé, je n’ai pas compris votre demande. Pouvez-vous préciser, comme lister nos produits ou vérifier une commande ?' "
            f"Sortie exactement dans ce format :\n"
            f"**Response:** message_clarification"
        ),
        "arabic": (
            f"أنت وكيل تجارة إلكترونية تساعد العملاء في استفسارات المنتجات، الطلبات، والمشكلات. "
            f"إدخال المستخدم '{user_input}' لم يمكن تصنيفه في نية محددة. "
            f"إنشاء رسالة ودية بالعربية تطلب من المستخدم توضيح طلبه. "
            f"اقترح خيارات مثل سرد المنتجات أو التحقق من طلب. "
            f"اجعلها قصيرة، طبيعية، واحترافية. "
            f"مثال: 'عذرًا، لم أفهم طلبك. هل يمكنك التوضيح، مثل طلب قائمة المنتجات أو التحقق من طلب؟' "
            f"الإخراج بالضبط بهذا الشكل:\n"
            f"**Response:** رسالة_توضيح"
        ),
    }

    prompt = clarification_templates.get(language, clarification_templates["english"])
    message = HumanMessage(content=prompt)

    try:
        logger.info(f"Streaming LLM response for clarification in {language}:")
        response = llm.invoke([message]).content
        # logger.info(f"Raw LLM response: {response}")

        clarification_message = extract_answer(response, "**Response:**")
        if not clarification_message:
            logger.warning(
                f"No clarification message extracted, using fallback for {language}"
            )
            fallback_messages = {
                "english": "Sorry, I didn’t understand your request. Could you clarify, like asking for our products or checking an order?",
                "french": "Désolé, je n’ai pas compris votre demande. Pouvez-vous préciser, comme lister nos produits ou vérifier une commande ?",
                "arabic": "عذرًا، لم أفهم طلبك. هل يمكنك التوضيح، مثل طلب قائمة المنتجات أو التحقق من طلب؟",
            }
            clarification_message = fallback_messages.get(
                language, fallback_messages["english"]
            )

        logger.info(f"Generated clarification in {language}: {clarification_message}")
        return {**state, "response": clarification_message}

    except Exception as e:
        logger.error(f"Error generating clarification: {str(e)}")
        fallback_messages = {
            "english": "Sorry, I didn’t understand your request. Could you clarify, like asking for our products or checking an order?",
            "french": "Désolé, je n’ai pas compris votre demande. Pouvez-vous préciser, comme lister nos produits ou vérifier une commande ?",
            "arabic": "عذرًا، لم أفهم طلبك. هل يمكنك التوضيح، مثل طلب قائمة المنتجات أو التحقق من طلب؟",
        }
        return {
            **state,
            "response": fallback_messages.get(language, fallback_messages["english"]),
        }


def handle_greeting(state: dict) -> dict:
    """
    Handle greeting intent and generate a response in the detected language.
    """
    user_input = state.get("user_input", "").strip()
    language = state.get("language", "english").lower()
    name = state.get("name", "User")

    # Define greeting templates by language with E-commerce context
    greeting_templates = {
        "english": (
            f"You are an E-commerce Agent assisting customers with product inquiries. "
            f"Generate a friendly greeting in English for {name} based on their input '{user_input}'. "
            f"Echo the greeting (e.g., 'hello' → 'Hello!') and offer assistance politely. "
            f"Keep it short, natural, and professional. "
            f"Example: If input is 'hello', respond 'Hello! How can I assist you today?' "
            f"Output exactly in this format:\n"
            f"**Response:** greeting_message"
        ),
        "french": (
            f"Vous êtes un Agent E-commerce aidant les clients avec des demandes sur les produits. "
            f"Générer une salutation amicale en français pour {name} basée sur l'entrée '{user_input}'. "
            f"Reprendre la salutation (par exemple, 'bonsoir' → 'Bonsoir !') et proposer de l'aide poliment. "
            f"Garder court, naturel et professionnel. "
            f"Exemple : Si l'entrée est 'bonsoir', répondre 'Bonsoir ! Comment puis-je vous aider aujourd'hui ?' "
            f"Sortie exactement dans ce format :\n"
            f"**Response:** message_de_salutation"
        ),
        "arabic": (
            f"أنت وكيل تجارة إلكترونية تساعد العملاء في استفسارات المنتجات. "
            f"إنشاء تحية ودية بالعربية لـ {name} بناءً على إدخالهم '{user_input}'. "
            f"كرر التحية (مثل 'مرحبا' → 'مرحبًا!') وعرض المساعدة بأدب. "
            f"اجعلها قصيرة، طبيعية، واحترافية. "
            f"مثال: إذا كان الإدخال 'مرحبا'، أجب 'مرحبًا! كيف يمكنني مساعدتك اليوم؟' "
            f"الإخراج بالضبط بهذا الشكل:\n"
            f"**Response:** رسالة_تحية"
        ),
    }

    prompt = greeting_templates.get(language, greeting_templates["english"])
    message = HumanMessage(content=prompt)

    try:
        logger.info(f"Streaming LLM response for greeting in {language}:")
        response = llm.invoke([message]).content
        # logger.info(f"Raw LLM response: {response}")

        greeting_message = extract_answer(response, "**Response:**")
        if not greeting_message:
            logger.warning(
                f"No greeting message extracted, using fallback for {language}"
            )
            fallback_messages = {
                "english": f"Hello! How can I assist you today?",
                "french": f"Bonsoir ! Comment puis-je vous aider aujourd'hui ?",
                "arabic": f"مرحبًا! كيف يمكنني مساعدتك اليوم؟",
            }
            greeting_message = fallback_messages.get(
                language, fallback_messages["english"]
            )

        logger.info(f"Generated greeting in {language}: {greeting_message}")
        return {**state, "response": greeting_message}

    except Exception as e:
        logger.error(f"Error generating greeting: {str(e)}")
        fallback_messages = {
            "english": f"Hello! How can I assist you today?",
            "french": f"Bonsoir ! Comment puis-je vous aider aujourd'hui ?",
            "arabic": f"مرحبًا! كيف يمكنني مساعدتك اليوم؟",
        }
        return {
            **state,
            "response": fallback_messages.get(language, fallback_messages["english"]),
        }


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
    print("state request_items:", type(requested_items))
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
            f"Available products: {', '.join([p['name'] for p in products])}. "
            f"Identify the most likely product for each requested item, accounting for misspellings or shortened names. "
            f"Match based on string similarity and context (e.g., 'Presse Argume' should match 'Presse Agrume Silver Crest YZI-001 45W Rose' due to shared 'Presse Agrume'). "
            f"Return product names from the list as a comma-separated string, in the same order as the requested items. "
            # f"Use 'none' for any item with no likely match. "
            f"Do not include extra products beyond the requested items. "
            f"Output exactly in this format:\n"
            f"**Products:** product_name_1,product_name_2,...,product_name_n or none"
        )
        message = HumanMessage(content=prompt)
        try:
            print(f"LLM call for product matching: {', '.join(requested_items)}")
            response = llm.invoke([message]).content
            print(f"LLM response: {response}")
            matched_products = extract_answer(response, "**Products:**")
            # if matched_products_raw and len(matched_products_raw) > 0:
            #     matched_products = [
            #         p.strip() for p in matched_products_raw if p.strip()
            #     ]
            if not matched_products or len(matched_products) != len(requested_items):
                logger.warning(
                    f"LLM returned invalid matches for {requested_items}: {matched_products}"
                )
                matched_products = ["none"] * len(requested_items)
            else:
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

    # Handle address
    if state.get("address", "none") != "none" and state["address"] != "unknown":
        try:
            api_call(
                "update_address", {"user_id": user_id, "address": state["address"]}
            )
            order_data = {"items": valid_products, "status": "Pending"}
            result = api_call(
                "new_order", {"user_id": user_id_converty, "order_data": order_data}
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
                    f"Include the delivery address '{state['address']}'. "
                    f"Keep it short and friendly. "
                    f"Example in English: 'Your order for {', '.join(valid_products)} has been confirmed (ID: {result['order_id']}). We’ll deliver to {state['address']}. Thank you!' "
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
            logger.error(f"Error in handle_new_order with provided address: {str(e)}")
            if language == "french":
                response = f"Une erreur s’est produite lors de la création de votre commande pour {', '.join(valid_products)}. Veuillez réessayer."
            elif language == "arabic":
                response = f"حدث خطأ أثناء إنشاء طلبك لـ {', '.join(valid_products)}. يرجى المحاولة مرة أخرى."
            else:
                response = f"An error occurred while creating your order for {', '.join(valid_products)}. Please try again."
            state["response"] = response
    elif existing_address != "none" and existing_address != "unknown":
        try:
            order_data = {"items": valid_products, "status": "Pending"}
            result = api_call(
                "new_order", {"user_id": user_id_converty, "order_data": order_data}
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


# def handle_address_input(state: AgentState, config: RunnableConfig) -> AgentState:
#     if (
#         state.get("next_step") != "await_address"
#         or state.get("intent") != "address_provider"
#     ):
#         logger.info(
#             f"Not expecting address yet or intent not 'address_provider' for {config['configurable']['phone_number']}, skipping handle_address_input"
#         )
#         items = state.get("requested_items", ["item"])
#         items_str = " and ".join(items) if len(items) > 1 else items[0]
#         prompt = (
#             f"Generate a message in {state['language']} informing the user that no address was detected and requesting a shipping address for {items_str} (e.g., street, city, country). "
#             f"Keep it short and friendly. "
#             f"Output exactly in this format:\n"
#             f"**Response:** address_request_message"
#         )
#         message = HumanMessage(content=prompt)
#         response = llm.invoke([message]).content
#         state["response"] = extract_answer(response, "**Response:**")
#         state["next_step"] = "await_address"
#         return state

#     user_id = config.get("configurable", {}).get("phone_number")
#     language = state["language"]
#     address = state["address"] or state["user_input"].strip()

#     try:
#         api_call("update_address", {"user_id": user_id, "address": address})
#         state["address"] = address
#         items = state.get("requested_items", [])
#         order_data = {"items": items, "status": "Pending"}
#         result = api_call(
#             "create_order", {"user_id": user_id, "order_data": order_data}
#         )
#         state["order_data"] = result
#         items_str = (
#             " and ".join(items) if len(items) > 1 else items[0] if items else "item"
#         )
#         prompt = (
#             f"Generate a confirmation message in {language} for an order of {items_str} with Order ID {result['order_id']}, noting that the address was updated. "
#             f"Keep it short and friendly. "
#             f"Output exactly in this format:\n"
#             f"**Response:** confirmation_message"
#         )
#         message = HumanMessage(content=prompt)
#         response = llm.invoke([message]).content
#         state["response"] = extract_answer(response, "**Response:**")
#         state["next_step"] = None
#         state["intent"] = None
#         state["order_data"] = None
#         state["requested_items"] = None
#     except Exception as e:
#         logger.error(f"Error in handle_address_input: {e}")
#         prompt = (
#             f"Generate an error message in {language} indicating an issue with the address: {str(e)}. "
#             f"Keep it short and friendly. "
#             f"Output exactly in this format:\n"
#             f"**Response:** error_message"
#         )
#         message = HumanMessage(content=prompt)
#         response = llm.invoke([message]).content
#         state["response"] = extract_answer(response, "**Response:**")
#     logger.info(f"State after handle_address_input: {state}")
#     return state


def handle_address_input(state: dict, config: dict) -> dict:
    """
    Handle user input as an address for a new order, update address, and create the order.
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
    requested_items = state.get("requested_items", "none")

    logger.info(
        f"Handling address input '{user_input}' for user_id: {user_id_converty}, requested_items: {requested_items}"
    )

    if not user_input:
        if language == "french":
            response = (
                f"Veuillez fournir une adresse valide pour commander {requested_items}."
            )
        elif language == "arabic":
            response = f"يرجى تقديم عنوان صالح لطلب {requested_items}."
        else:
            response = f"Please provide a valid address to order {requested_items}."
        state["response"] = response
        state["next_step"] = "await_address"
        logger.info(f"Generated response for empty address: {response}")
        return state

    if requested_items == "none":
        if language == "french":
            response = "Erreur : aucun produit sélectionné. Veuillez indiquer le produit que vous souhaitez commander."
        elif language == "arabic":
            response = "خطأ: لم يتم اختيار أي منتج. يرجى تحديد المنتج الذي تريد طلبه."
        else:
            response = "Error: No product selected. Please specify the product you want to order."
        state["response"] = response
        state["next_step"] = None
        logger.info(f"Generated response for missing product: {response}")
        return state

    try:
        # Update address
        api_call("update_address", {"user_id": user_id, "address": user_input})
        # Create order
        order_data = {"items": [requested_items], "status": "Pending"}
        result = api_call(
            "new_order", {"user_id": user_id_converty, "order_data": order_data}
        )
        if "error" in result:
            logger.error(f"Failed to create order: {result['error']}")
            if language == "french":
                response = f"Une erreur s’est produite lors de la création de votre commande pour {requested_items}. Veuillez réessayer."
            elif language == "arabic":
                response = f"حدث خطأ أثناء إنشاء طلبك لـ {requested_items}. يرجى المحاولة مرة أخرى."
            else:
                response = f"An error occurred while creating your order for {requested_items}. Please try again."
        else:
            state["order_data"] = result
            prompt = (
                f"Generate a confirmation message in {language} for an order of {requested_items} with Order ID {result['order_id']}. "
                f"Include the delivery address '{user_input}'. "
                f"Keep it short and friendly. "
                f"Example in English: 'Your order for {requested_items} has been confirmed (ID: {result['order_id']}). We’ll deliver to {user_input}. Thank you!' "
                f"Output exactly in this format:\n"
                f"**Response:** confirmation_message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
            state["next_step"] = None
            state["requested_items"] = None
            state["address"] = None
    except Exception as e:
        logger.error(f"Error in handle_address_input: {str(e)}")
        if language == "french":
            response = f"Une erreur s’est produite lors de la création de votre commande pour {requested_items}. Veuillez réessayer."
        elif language == "arabic":
            response = f"حدث خطأ أثناء إنشاء طلبك لـ {requested_items}. يرجى المحاولة مرة أخرى."
        else:
            response = f"An error occurred while creating your order for {requested_items}. Please try again."
        state["response"] = response

    logger.info(f"State after handle_address_input: {state}")
    return state


def handle_report_issue(state: AgentState, config: RunnableConfig) -> AgentState:
    user_id = config.get("configurable", {}).get("phone_number")
    language = state["language"]
    name = config.get("configurable", {}).get("name")
    issue_product = state.get("issue_product")
    user_input = state["user_input"]

    if not issue_product:
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
        orders = api_call("get_orders", {"user_id": user_id})
        has_ordered = any(
            issue_product.lower() in [item.lower() for item in order.get("items", [])]
            for order in orders
        )

        if has_ordered:
            issue_data = {
                "product": issue_product,
                "description": user_input,
                "name": name,
                "phone_number": user_id,
                "status": "Pending",
            }
            result = api_call("save_issue", {"user_id": user_id, "issue": issue_data})
            prompt = (
                f"Generate a message in {language} thanking the user for reporting an issue with {issue_product} and informing them an agent will contact them soon. Include Issue ID: {result['issue_id']}. "
                f"Keep it short and friendly. "
                f"Output exactly in this format:\n"
                f"**Response:** message"
            )
            message = HumanMessage(content=prompt)
            response = llm.invoke([message]).content
            state["response"] = extract_answer(response, "**Response:**")
        else:
            prompt = (
                f"Generate a message in {language} informing the user that they haven’t ordered a {issue_product} and asking them to specify a purchased product (e.g., 'problem with my phone'). "
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


def retrieve_order(state: AgentState, config: RunnableConfig) -> AgentState:
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


def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
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
