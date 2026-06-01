from redis import Redis
from httpx import AsyncClient
import asyncio
import os
import json
from pydantic import BaseModel


class Pipeline:
    """Пайплайн фильтра памяти: сохранение и инъекция фактов о пользователе."""

    class Valves(BaseModel):
        """Конфигурационные параметры пайплайна."""
        pipelines: list = ["*"]
        REDIS_HOST: str = os.getenv("REDIS_HOST", "hackathon-redis")
        REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
        MEMORY_TTL: int = 60 * 60 * 24 * 30
        MAX_FACTS: int = 20

    def __init__(self):
        """Инициализация пайплайна."""
        self.name = "Memory Filter"
        self.valves = self.Valves()
        self.type = "filter"
        self.redis = None

    async def on_startup(self):
        """Подключение к Redis при запуске пайплайна."""
        self.redis = Redis(host=self.valves.REDIS_HOST,
                           port=self.valves.REDIS_PORT,
                           decode_responses=True)
        print("Memory pipeline started", flush=True)

    async def on_shutdown(self):
        """Закрытие соединения с Redis при остановке пайплайна."""
        if self.redis:
            self.redis.close()
        print("Memory pipeline stopped", flush=True)

    async def inlet(self, body: dict, user: dict) -> dict:
        """Инъекция сохранённых фактов о пользователе в системный промпт перед
           запросом к модели."""
        user_id = user.get("id", "default")
        print(f"Memory inlet for user: {user_id}", flush=True)

        facts = self.get_facts(user_id)

        if facts:
            memory_text = "Что ты знаешь о пользователе:\n" + \
                "\n".join(f"- {f}" for f in facts)

            messages = body.get("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{memory_text}"
            else:
                messages.insert(0, {"role": "system", "content": memory_text})

            body["messages"] = messages
            print(
                f"Injected {len(facts)} facts for user {user_id}", flush=True)

        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        """Извлечение и сохранение новых фактов из сообщения пользователя
           после ответа модели."""
        user_id = user.get("id", "default")

        messages = body.get("messages", [])

        user_messages = [m for m in messages if m["role"] == "user"]
        if user_messages:
            last_user_message = user_messages[-1].get("content", "")
            await self.extract_and_save(user_id, last_user_message)

        return body

    def get_facts(self, user_id: str) -> list:
        """Получение списка сохранённых фактов для пользователя из Redis."""
        if not self.redis:
            return []
        try:
            facts = self.redis.lrange(f"memory:{user_id}", 0, -1)
            return facts
        except Exception as e:
            print(f"Redis read error: {e}", flush=True)
            return []

    def save_fact(self, user_id: str, fact: str):
        """Сохранение нового факта для пользователя в Redis с ограничением
           по количеству и TTL."""
        if not self.redis:
            return
        try:
            key = f"memory:{user_id}"
            existing = self.redis.lrange(key, 0, -1)
            if fact in existing:
                return
            self.redis.lpush(key, fact)
            self.redis.ltrim(key, 0, self.valves.MAX_FACTS - 1)
            self.redis.expire(key, self.valves.MEMORY_TTL)
            print(f"Saved fact for {user_id}: {fact}", flush=True)
        except Exception as e:
            print(f"Redis write error: {e}", flush=True)

    async def extract_and_save(self, user_id: str, message: str):
        """Извлечение фактов из сообщения пользователя через LLM 
           и их сохранение."""
        if len(message) < 10:
            return

        api_key = os.getenv("MWS_API_KEY")
        if not api_key:
            print("MWS_API_KEY not set, skipping fact extraction", flush=True)
            return

        base_url = os.getenv("MWS_BASE_URL", "https://api.gpt.mws.ru/v1")

        prompt = """Извлеки факты о пользователе из сообщения.
                Факты — это личная информация: имя, профессия, город, школа/вуз, интересы, предпочтения.
                ВАЖНО: Если в сообщении нет конкретной личной информации, ответь ТОЛЬКО ПУСТЫМ МАССИВОМ [].
                ЗАПРЕЩЕНО писать "Не указано", "Неизвестно" или какие-либо пояснения. Отвечай только фактами, которые явно назвал пользователь.
                Формат ответа: ТОЛЬКО валидный JSON массив строк.

                Пример 1:
                Сообщение: "меня зовут Сева, я учусь в МЭИ на третьем курсе и обожаю пить кофе"
                Ответ: ["Пользователя зовут Сева", "Учится в МЭИ на 3 курсе", "Любит пить кофе"]

                Пример 2:
                Сообщение: "как дела?"
                Ответ: []

                Пример 3:
                Сообщение: "куда поступать в магистратуру?"
                Ответ: []

                Сообщение: "{message}"
                Ответ:""".format(message=message[:500])

        try:
            async with AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-oss-20b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0,
                    }
                )
            content = response.json().get("choices", [{}])[0].get(
                "message", {}).get("content", "") or ""
            content = content.strip()
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                content = content[start:end+1]

            facts = json.loads(content)
            for fact in facts:
                if isinstance(fact, str) and fact:
                    self.save_fact(user_id, fact)
        except Exception as e:
            print(f"Extract facts error: {e}", flush=True)
