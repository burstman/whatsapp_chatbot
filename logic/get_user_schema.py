from sqlalchemy import inspect
from .table_db_logic import SessionLocal, User
from .agent_state import AgentState, RunnableConfig


def get_database_schema(engine):
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name}: {col_type}\n"
        schema += "\n"
    print("Retrieved database schema.")
    return schema


def get_current_user(state: dict, config: RunnableConfig) -> AgentState:
    print("Retrieving the current user based on user ID.")

    # Access the configurable field directly from RunnableConfig
    configurable = config.get("configurable", {})
    print(f"Configurable: {configurable}")

    # Try to get user_id from config
    user_id = configurable.get("current_user_id", None)

    if user_id is None and "current_user_id" in state:
        user_id = state["current_user_id"]
        print("Using state to get the current user ID from the state.")

    print(f"User ID: {user_id}")

    if not isinstance(state, dict):
        raise TypeError(
            f"Expected state to be a dictionary in get_current_user, but got {type(state).__name__}"
        )

    state_obj = AgentState(**state)

    if not user_id:
        print("No user ID provided in the configuration.")
        return AgentState(**state_obj.model_dump() | {"current_user": "User not found"})

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == int(user_id)).first()
        print(f"Query result: {user}")
        if user:
            current_user = user.name
            print(f"Current user set to: {current_user}")
            return AgentState(**state_obj.model_dump() | {"current_user": current_user})
        else:
            print("User not found in the database.")
            return AgentState(
                **state_obj.model_dump() | {"current_user": "User not found"}
            )
    except Exception as e:
        print(f"Error retrieving user: {str(e)}")
        return AgentState(
            **state_obj.model_dump() | {"current_user": "Error retrieving user"}
        )
    finally:
        session.close()
