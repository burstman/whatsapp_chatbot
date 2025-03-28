#!/usr/bin/env python3
import os
import sys
import logging
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    Float,
    DateTime,
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone = Column(String, unique=True, index=True)  # WhatsApp phone numbers as strings
    city = Column(String, nullable=True)
    address = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)

    orders = relationship("Order", back_populates="user")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    price = Column(Float)
    category = Column(String, index=True)

    orders = relationship("Order", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)  # Added for order history

    user = relationship("User", back_populates="orders")
    product = relationship("Product", back_populates="orders")


DATABASE_URL = "sqlite:///ecommerce.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(reset=False):
    """Initialize the database with sample data for an e-commerce business."""
    if reset:
        logger.info("Resetting database: dropping all tables and recreating them.")
        Base.metadata.drop_all(bind=engine)
    else:
        if os.path.exists("ecommerce.db"):
            logger.info(
                "Database 'ecommerce.db' already exists. Use --reset to recreate it."
            )
            return

    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

    with SessionLocal() as session:
        try:
            # Add users (with WhatsApp-like phone numbers)
            users = [
                User(
                    name="Alice",
                    phone="1234567890",
                    city="New York",
                    address="123 Main St",
                    email="alice@example.com",
                ),
                User(
                    name="Bob",
                    phone="2345678901",
                    city="Boston",
                    address="456 Oak St",
                    email="bob@example.com",
                ),
                User(
                    name="Charlie",
                    phone="3456789012",
                    city="Chicago",
                    address="789 Pine St",
                    email="charlie@example.com",
                ),
            ]
            session.add_all(users)
            session.commit()
            logger.info("Sample users added to the database.")

            # Add e-commerce products (reverted to your original example)
            products = [
                Product(name="Boite lunch Box", price=18, category="Kitchen"),
                Product(
                    name="Solar interaction wall lamp", price=23, category="Lighting"
                ),
                Product(
                    name="Presse Agrume Silver Crest", price=38, category="Kitchen"
                ),
            ]
            session.add_all(products)
            session.commit()
            logger.info("Sample products added to the database.")

            # Add sample orders
            orders = [
                Order(product_id=1, user_id=1, created_at=datetime.utcnow()),
                Order(product_id=2, user_id=1, created_at=datetime.utcnow()),
                Order(product_id=3, user_id=2, created_at=datetime.utcnow()),
            ]
            session.add_all(orders)
            session.commit()
            logger.info("Sample orders added to the database.")

            # Verify and print the inserted data
            print("\nDatabase Contents:")
            print("\nUsers:")
            for user in session.query(User).all():
                print(
                    f"ID: {user.id}, Name: {user.name}, Phone: {user.phone}, Email: {user.email}"
                )

            print("\nProducts:")
            for product in session.query(Product).all():
                print(
                    f"ID: {product.id}, Name: {product.name}, Price: {product.price}, Category: {product.category}"
                )

            print("\nOrders:")
            for order in session.query(Order).all():
                print(
                    f"ID: {order.id}, User ID: {order.user_id}, Product ID: {order.product_id}, Created At: {order.created_at}"
                )

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    logger.info("Database successfully initialized and filled with sample data.")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    init_db(reset=reset)
