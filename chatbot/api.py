# Directory: llm_chatbot_server/chatbot/
# Purpose: Handles interactions with the Converty API and database for user, order, and issue operations.
# Prerequisites:
#   - Dependencies: sqlalchemy, converty-api (custom or installed package).
#   - chatbot/db.py for SessionLocal, User, Claim.
# Notes:
#   - Integrates with Converty API for product listing, order creation, and retrieval (per April 17, 2025).
#   - If converty-api is custom, ensure it's in PYTHONPATH or installed.
#   - Fallback product list included for API failures.

import logging
from chatbot.db import SessionLocal, User, Claim
from api.converty import (
    get_customer_orders,
    CustomerOrderQuery,
    create_order,
    Customer,
    CartItem,
    map_product_name_to_id,
    fetch_converty_products,
)

logger = logging.getLogger(__name__)


def api_call(endpoint: str, payload: dict = None):
    """Handle API calls for user, order, and issue operations using SQLAlchemy and Converty API."""
    with SessionLocal() as session:
        try:
            if endpoint == "get_user":
                user_id = payload["user_id"]
                name = payload.get("name", "Unknown User")
                user = session.query(User).filter_by(phone_number=user_id).first()
                if user:
                    return {"name": user.name, "address": user.address or "unknown"}
                else:
                    new_user = User(phone_number=user_id, name=name, address="unknown")
                    session.add(new_user)
                    session.commit()
                    logger.info(
                        f"Created new user with phone {user_id} and name {name}"
                    )
                    return {"name": name, "address": "unknown"}

            elif endpoint == "get_orders":
                user_id = payload["user_id"]
                try:
                    query = CustomerOrderQuery(page=1, limit=10, status="pending")
                    orders = get_customer_orders(query)
                    formatted_orders = [
                        {
                            "order_id": order["_id"],
                            "items": [
                                item["product"]["name"]
                                for item in order.get("cart", [])
                            ],
                            "status": order["status"],
                        }
                        for order in orders
                    ]
                    return formatted_orders
                except Exception as e:
                    logger.error(f"Error fetching orders from Converty API: {e}")
                    return []

            elif endpoint == "list_products":
                try:
                    products = fetch_converty_products()
                    formatted_products = [
                        {
                            "id": product.get("_id", f"p{index+1}"),
                            "name": product.get("name", "Unknown Product"),
                            "price": product.get("price", 0.0),
                        }
                        for index, product in enumerate(products)
                    ]
                    return formatted_products
                except Exception as e:
                    logger.error(f"Error fetching products from Converty API: {e}")
                    return []

            elif endpoint == "new_order":
                user_id = payload["user_id"]
                order_data = payload["order_data"]
                items = order_data["items"]
                address = payload.get("address", "unknown")
                try:
                    cart_items = [
                        CartItem(product_id=map_product_name_to_id(item), quantity=1)
                        for item in items
                    ]
                except ValueError as e:
                    logger.error(f"Error mapping product names: {e}")
                    return {"error": str(e)}

                user = session.query(User).filter_by(phone_number=user_id).first()
                if not user:
                    raise ValueError("User not found")

                customer = Customer(
                    name=user.name or "Unknown", phone=user_id, address=address
                )

                try:
                    order_result = create_order(
                        customer=customer, cart_items=cart_items, status="pending"
                    )
                    order_id = order_result.get("_id")
                    if not order_id:
                        raise ValueError("Order creation failed: No order ID returned")

                    claim_details = {
                        "order_id": order_id,
                        "items": items,
                        "product_ids": [item.product_id for item in cart_items],
                        "status": "pending",
                    }
                    claim = Claim(
                        user_id=user.id,
                        type="order",
                        details=claim_details,
                        status="pending",
                    )
                    session.add(claim)
                    session.commit()
                    return {"status": "success", "order_id": f"ord{claim.id}"}
                except Exception as e:
                    logger.error(f"Error creating order in Converty API: {e}")
                    return {"error": str(e)}

            elif endpoint == "update_address":
                user_id = payload["user_id"]
                address = payload["address"]
                user = session.query(User).filter_by(phone_number=user_id).first()
                if not user:
                    raise ValueError("User not found")
                user.address = address
                claim = Claim(
                    user_id=user.id,
                    type="address",
                    details={"address": address},
                    status="completed",
                )
                session.add(claim)
                session.commit()
                return {"status": "address_updated"}

            elif endpoint == "save_issue":
                user_id = payload["user_id"]
                issue = payload["issue"]
                user = session.query(User).filter_by(phone_number=user_id).first()
                if not user:
                    raise ValueError("User not found")
                claim = Claim(
                    user_id=user.id, type="issue", details=issue, status="pending"
                )
                session.add(claim)
                session.commit()
                return {"status": "success", "issue_id": f"iss{claim.id}"}

            return {"error": "Invalid endpoint"}

        except Exception as e:
            session.rollback()
            logger.error(f"Error in api_call for endpoint {endpoint}: {e}")
            return {"error": str(e)}
