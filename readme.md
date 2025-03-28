```
ecommerce_chatbot/
├── database/
│   ├── __init__.py         # Marks directory as a package
│   ├── models.py           # SQLAlchemy models (User, Product)
│   └── db.py               # Database setup (engine, session)
├── chatbot/
│   ├── __init__.py         # Exports all nodes
│   ├── state.py            # AgentState Pydantic model
│   ├── extract.py          # extract_product_items function
│   ├── check.py            # check_product_existence function
│   ├── generate.py         # generate_human_readable_answer function
│   └── utils.py            # format_product_list helper
├── api/
│   ├── __init__.py         # Marks directory as a package
│   └── main.py             # FastAPI app (endpoints)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```