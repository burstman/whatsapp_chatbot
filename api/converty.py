import requests
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


# Converty API endpoint
API_URL = "https://api.converty.shop/api/v1/products"
ORDERS_API_URL = "https://api.converty.shop/api/v1/orders"
# LocalApi
REFRESH_TOKEN_URL = "http://localhost:9001/GetAccessToken"


def get_valid_access_token():
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(REFRESH_TOKEN_URL, headers=headers)
        response.raise_for_status()

        data = response.json()
        if "access_token" not in data:
            raise Exception("No access token returned in refresh response")

        return data["access_token"]
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to token refresh server")
    except requests.exceptions.Timeout:
        raise Exception("Token refresh request timed out")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error during token refresh: {e}")
    except ValueError:
        raise Exception("Invalid JSON response from token refresh server")


def fetch_converty_products():
    """
    Fetch products from the Converty API with robust token handling.
    
    This function retrieves product data from the Converty shop API, with built-in error handling
    for authentication and request issues. It supports automatic token refresh on 401 Unauthorized
    responses and validates the API response.
    
    Returns:
        List[Dict]: A list of product dictionaries from the Converty API.
    
    Raises:
        Exception: If there are issues with token retrieval, API authentication, or request processing.
    """
    try:
        access_token = get_valid_access_token()

        # Make the API request to Converty
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(API_URL, headers=headers)
            # Check for 401 Unauthorized
            if response.status_code == 401:
                print("Received 401 Unauthorized, attempting to refresh token...")
                new_token = get_valid_access_token()
                if new_token:
                    headers["Authorization"] = f"Bearer {new_token}"
                    response = requests.get(API_URL, headers=headers)
                else:
                    raise Exception("Token refresh failed after 401 response")

            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            # Check if the response indicates success
            if not data.get("success", False):
                raise Exception(
                    f"API request failed: {data.get('message', 'Unknown error')}"
                )

            # Extract and return the products
            return data.get("data", [])

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("Failed to authenticate with API: 401 Unauthorized")
            else:
                print(f"HTTP error making API request: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            raise
    except Exception as e:
        print(f"Error getting access token: {e}")
        raise


def map_product_name_to_id(product_name: str) -> str:
    """Map a product name to its ID by fetching products from the Converty shop API.

    Args:
        product_name: The name of the product to find (case-insensitive).

    Returns:
        The product ID (str) corresponding to the product name.

    Raises:
        ValueError: If the product name is not found or multiple products match.
    """
    products = fetch_converty_products()
    if not products:
        raise ValueError("No products available to map")

    # Create a case-insensitive mapping of names to IDs
    name_to_id = {}
    for product in products:
        name = product.get("name", "").lower()
        product_id = product.get("_id")
        if name and product_id:
            if name in name_to_id:
                raise ValueError(f"Multiple products found with name '{product_name}'")
            name_to_id[name] = product_id

    # Look up the product name (case-insensitive)
    product_name_lower = product_name.lower()
    product_id = name_to_id.get(product_name_lower)
    if not product_id:
        raise ValueError(f"Product name '{product_name}' not found")
    print(f"Mapped product '{product_name}' to ID: {product_id}")
    return product_id


@dataclass
class CartItem:
    product_id: str
    quantity: int
    selected_variants: Optional[List[Dict]] = None


@dataclass
class Customer:
    name: str
    phone: str
    address: Optional[str] = None
    note: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None


def create_order_payload(
    customer: Customer,
    cart_items: List[CartItem],
    status: str = "pending",
    attempt: int = 0,
    note: Optional[str] = None,
) -> Dict:
    """Create and validate the order payload with prices fetched from products API."""
    # Validate required fields
    if not customer.phone:
        raise ValueError("Customer phone number is required")
    if not cart_items:
        raise ValueError("Cart cannot be empty")

    for item in cart_items:
        if not item.product_id:
            raise ValueError("Product ID is required for each cart item")
        if item.quantity < 1:
            raise ValueError("Quantity must be at least 1 for each cart item")

    # Fetch products to get prices
    products = fetch_converty_products()
    product_map = {p["_id"]: p for p in products}

    # Build cart with fetched prices
    cart = []
    total_cart_price = 0.0
    delivery_cost = 0.0

    for item in cart_items:
        product = product_map.get(item.product_id)
        if not product:
            raise ValueError(f"Product ID {item.product_id} not found in products")

        price_per_unit = float(product.get("price", 0))
        if price_per_unit <= 0:
            raise ValueError(f"Invalid price for product ID {item.product_id}")

        item_delivery_cost = float(product.get("deliveryCost", 7.0))
        delivery_cost = max(
            delivery_cost, item_delivery_cost
        )  # Use highest delivery cost

        cart.append(
            {
                "product": item.product_id,
                "quantity": item.quantity,
                "pricePerUnit": price_per_unit,
                "selectedVariants": item.selected_variants or [],
            }
        )
        total_cart_price += price_per_unit * item.quantity

    # Calculate total price
    total_price = total_cart_price + delivery_cost

    # Build payload
    payload = {
        "status": status,
        "attempt": attempt,
        "total": {
            "deliveryPrice": delivery_cost,
            "deliveryCost": delivery_cost,
            "totalPrice": total_price,
        },
        "customer": {
            "name": customer.name.strip(),
            "phone": customer.phone.strip(),
            "address": customer.address.strip() if customer.address else "",
            "note": customer.note.strip() if customer.note else "",
            "email": customer.email.strip() if customer.email else "",
            "city": customer.city.strip() if customer.city else "",
        },
        "cart": cart,
    }

    if note:
        payload["note"] = note.strip()

    print(f"Generated order Payload: {json.dumps(payload,indent=4)}")

    return payload


def get_all_product_names() -> list[str]:
    """Get the names of all products from the Converty shop API."""
    products = fetch_converty_products()
    return [product.get("name", "") for product in products]


def create_order(
    customer: Customer,
    cart_items: List[CartItem],
    status: str = "pending",
    attempt: int = 0,
    note: Optional[str] = None,
) -> Dict:
    """Create an order in the Converty shop."""
    access_token = get_valid_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Create payload
    order_payload = create_order_payload(
        customer=customer,
        cart_items=cart_items,
        status=status,
        attempt=attempt,
        note=note,
    )

    try:
        response = requests.post(ORDERS_API_URL, headers=headers, json=order_payload)
        if response.status_code == 401:
            print("Received 401 Unauthorized, attempting to refresh token...")
            new_token = get_valid_access_token()
            if new_token:
                headers["Authorization"] = f"Bearer {new_token}"
                response = requests.post(
                    ORDERS_API_URL, headers=headers, json=order_payload
                )
            else:
                raise Exception("Token refresh failed after 401 response")

        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            raise Exception(
                f"Order creation failed: {data.get('message', 'Unknown error')}"
            )

        return data.get("data", {})

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Failed to authenticate with API: 401 Unauthorized")
        else:
            print(f"HTTP error creating order: {e}")
            print(f"response: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error creating order: {e}")
        raise


@dataclass
class CustomerOrderQuery:
    """Query parameters for fetching customer orders."""

    page: int = 1
    limit: int = 10
    status: Optional[str] = None
    archived: Optional[bool] = None
    abandoned: Optional[bool] = None
    deleted: Optional[bool] = None
    search: Optional[str] = None
    product: Optional[str] = None
    delivery_company: Optional[str] = None

    def __post_init__(self):
        """Validate query parameters."""
        if self.page < 1:
            raise ValueError("Page number must be at least 1")
        if self.limit < 1:
            raise ValueError("Limit must be at least 1")


def get_customer_orders(query: CustomerOrderQuery = CustomerOrderQuery()) -> List[dict]:
    """Fetch customer orders for the specified store using query parameters.

    Args:
        query: A CustomerOrderQuery instance containing query parameters.

    Returns:
        A list of order dictionaries containing customer and order details.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    access_token = get_valid_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Build query parameters
    params = {"page": query.page, "limit": query.limit}
    if query.status:
        params["status"] = query.status
    if query.archived is not None:
        params["archived"] = "true" if query.archived else "false"
    if query.abandoned is not None:
        params["abandoned"] = "true" if query.abandoned else "false"
    if query.deleted is not None:
        params["deleted"] = "true" if query.deleted else "false"
    if query.search:
        params["search"] = query.search
    if query.product:
        params["product"] = query.product
    if query.delivery_company:
        params["deliveryCompany"] = query.delivery_company

    try:
        response = requests.get(ORDERS_API_URL, headers=headers, params=params)
        if response.status_code == 401:
            print("Received 401 Unauthorized, attempting to refresh token...")
            new_token = get_valid_access_token()
            if new_token:
                headers["Authorization"] = f"Bearer {new_token}"
                response = requests.get(ORDERS_API_URL, headers=headers, params=params)
            else:
                raise Exception("Token refresh failed after 401 response")

        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            raise Exception(
                f"Failed to fetch orders: {data.get('message', 'Unknown error')}"
            )

        return data.get("data", [])

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Failed to authenticate with API: 401 Unauthorized")
        else:
            print(f"HTTP error fetching orders: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error fetching orders: {e}")
        raise
