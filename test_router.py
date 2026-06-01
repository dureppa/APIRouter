import asyncio
from pipelines.router import Pipeline


async def test_semantic_routing():
    print("Initializing Pipeline...")
    p = Pipeline()
    await p.on_startup()

    test_queries = [
        "Нарисуй мне красивый пейзаж с горами",
        "Какая сегодня погода в Москве?",
        "Помоги решить сложную математическую задачу по физике",
        "Привет! Как твои дела?",
        "Сгенерируй изображение кота",
        "Найди в интернете последние новости про AI",
        "Проанализируй этот длинный текст и сделай выводы",
        "Что такое черная дыра?"
    ]

    print("\n--- Testing Semantic Router ---")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        intent = await p.semantic_classify(q)
        print(f"Result Intent: {intent}")

if __name__ == "__main__":
    asyncio.run(test_semantic_routing())
