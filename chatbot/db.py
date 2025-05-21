# Directory: llm_chatbot_server/chatbot/
# Purpose: Handles database setup, schema creation, and SQLAlchemy models (User, Claim) for the chatbot.
# Prerequisites:
#   - Environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT.
#   - Dependencies: sqlalchemy, psycopg2-binary, python-dotenv.
#   - PostgreSQL database converty_oauth with converty_user having CREATE privileges.
# Notes:
#   - Creates the chatbot schema and grants privileges to converty_user.
#   - Fixes ObjectNotExecutableError using sqlalchemy.sql.text() for schema creation.
#   - Run psql commands if schema creation fails:
#     psql -U postgres -h <DB_HOST> -p <DB_PORT> -d converty_oauth
#     CREATE SCHEMA IF NOT EXISTS chatbot;
#     GRANT ALL ON SCHEMA chatbot TO converty_user;

import os
import logging
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text, func
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

# Validate environment variables
for var, name in [
    (DB_HOST, "DB_HOST"),
    (DB_NAME, "DB_NAME"),
    (DB_USER, "DB_USER"),
    (DB_PASSWORD, "DB_PASSWORD"),
]:
    if not var:
        raise ValueError(f"{name} environment variable is not set")

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"DATABASE_URL: {DATABASE_URL.replace(DB_PASSWORD, '****')}")
engine = create_engine(DATABASE_URL)

# Create chatbot schema if it doesn't exist
try:
    with engine.connect() as connection:
        with connection.begin():  # Start a transaction
            # Create schema
            connection.execute(text("CREATE SCHEMA IF NOT EXISTS chatbot"))
            # Grant privileges
            connection.execute(text("GRANT ALL ON SCHEMA chatbot TO converty_user"))
            # Verify schema existence
            result = connection.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'chatbot'"
                )
            ).fetchone()
            if result:
                logger.info("Chatbot schema exists or was created successfully")
            else:
                logger.error("Chatbot schema creation failed")
                raise RuntimeError("Failed to create chatbot schema")
        logger.info("Granted privileges to converty_user on chatbot schema")
except Exception as e:
    logger.error(f"Error creating chatbot schema: {str(e)}")
    logger.error(traceback.format_exc())
    raise
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = {"schema": "chatbot"}
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("chatbot.users.id"), nullable=False)
    type = Column(String(20), nullable=False)
    details = Column(JSONB, nullable=False)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, server_default=func.now())
    user = relationship("User", back_populates="interactions")


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "chatbot"}
    id = Column(Integer, primary_key=True)
    phone_number = Column(String(50), unique=True, nullable=False)
    name = Column(String(100))
    address = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    interactions = relationship("Interaction", back_populates="user")


# Enable SQLAlchemy query logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
