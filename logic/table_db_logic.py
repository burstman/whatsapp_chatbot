from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import os


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ecommerce.db")
print(f"Using database URL: {DATABASE_URL}")
Engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone = Column(Integer, unique=True, index=True)
    city = Column(String)
    address = Column(String)
    email = Column(String, unique=True, index=True)

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
    product_id = Column(
        Integer, ForeignKey("products.id")
    )  # Changed from food_id to product_id
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="orders")
    product = relationship(
        "Product", back_populates="orders"
    )  # Fixed "Products" to "Product"
