from chatbot.llm import extract_answer

response = """<think>
test
</think>
**Response:** السلام عليكم. كيف يمكنني مساعدتك اليوم؟."""

language = extract_answer(response, "**Response:**").lower()

print(language)
