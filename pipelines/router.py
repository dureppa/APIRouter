from pydantic import BaseModel
import httpx
import logging
import os
import asyncio
from typing import Generator
import json


class Pipeline:
    """Роутер запросов к различным AI-моделям на основе типа задачи."""

    MWS_API_KEY: str = os.getenv("MWS_API_KEY", "")

    class Valves(BaseModel):
        """Конфигурационные параметры роутера."""
        MWS_BASE_URL: str = os.getenv(
            "MWS_BASE_URL", "https://api.gpt.mws.ru/v1")

        FORCE_REASONING: bool = os.getenv(
            "FORCE_REASONING", "false").lower() == "true"

        MODEL_CHAT: str = "gpt-oss-20b"
        MODEL_CLASSIFIER: str = "llama-3.1-8b-instruct"
        MODEL_VISION: str = "cotype-pro-vl-32b"
        MODEL_REASONING: str = "deepseek-r1-distill-qwen-32b"
        MODEL_AUDIO: str = 'whisper-turbo-local'
        MODEL_IMAGE_GEN: str = 'qwen-image'
        MODEL_EMBEDDING: str = 'bge-m3'

    def __init__(self):
        """Инициализация роутера."""
        self.name = "AI Router"
        self.valves = self.Valves()
        self.anchor_embeddings = {}

    async def on_startup(self):
        """Инициализация при запуске пайплайна."""
        print(f"Router pipeline started", flush=True)
        print(f"API KEY set: {bool(self.MWS_API_KEY)}", flush=True)
        if self.MWS_API_KEY:
            print(f"API KEY prefix: {self.MWS_API_KEY[:8]}...", flush=True)
        await self.warmup_embeddings()

    async def warmup_embeddings(self):
        """Инициализация эмбеддингов для интентов с кэшированием."""
        anchors = {
            "IMAGE_GEN": [
                "Нарисуй картинку", "Сгенерируй изображение", "Создай фото", "нарисуй", "сделай арт",
                "сбацай арт", "скинь фотографию", "покажи фото", "нарисуй эпичный арт", "создай картинку",
                "картинка", "изобрази", "набросай эскиз", "сгенерируй пикчу", "сделай иллюстрацию",
                "нарисуй мне", "фотореалистичное изображение", "сгенерируй обои", "нарисуй портрет"
            ],
            "VISION": [
                "Опиши картинку", "Что на картинке", "что изображено", "Проанализируй изображение",
                "Что на фото", "распознай текст на картинке", "что ты видишь здесь", "опиши это фото",
                "что происходит на картинке", "расшифруй изображение", "извлеки данные с фото"
            ],
            "SEARCH": [
                "Найди в интернете информацию", "Какая погода", "Узнай последние новости", "Поиск в сети",
                "Актуальный курс валют", "Кто такой", "Что сейчас происходит", "Текущая цена",
                "найди в интернете", "какой курс доллара к тенге", "покажи как выглядит", "найди реальное фото",
                "как выглядит сегодня", "найди фотографию в сети", "поищи картинку", "кто выиграл матч",
                "новости за сегодня", "загугли", "поищи в гугле", "найди свежую информацию", "какой сегодня праздник"
            ],
            "REASONING": [
                "Проанализируй данные", "Реши сложную задачу", "Подумай шаг за шагом", "Математическое доказательство",
                "реши уравнение", "докажи теорему", "логическая загадка", "проведи сложный анализ",
                "выведи формулу", "посчитай вероятность", "реши алгоритмическую задачу"
            ],
            "DEEP_RESEARCH": [
                "сделай глубокий анализ", "проведи исследование", "подробный ресерч", "глубокое исследование",
                "deep research", "собери всю информацию", "проанализируй рынок", "подготовь подробный отчет",
                "проведи глубокий поиск", "масштабный поиск", "напиши аналитический отчет"
            ],
            "WEB": [
                "Перейди по ссылке", "Опиши сайт", "Что находится на странице", "Открой сайт", "Прочитай по ссылке",
                "проанализируй ссылку", "содержимое сайта", "как написано на сайте", "что пишут по этому адресу",
                "сделай выжимку статьи", "проанализируй веб-страницу"
            ],
            "FILE": [
                "Проанализируй документ", "Прочитай файл", "Что в этом документе", "Краткое содержание файла",
                "Сумаризуй текст", "сделай саммари документа", "вытащи из файла", "расскажи про этот файл",
                "сделай выжимку из прикрепленного", "кратко перескажи текст", "о чем эта pdf-ка"
            ],
            "CHAT": [
                "Привет, как дела", "Напиши текст", "Помоги с кодом", "сгенерируй код", "напиши код",
                "Расскажи историю", "Что ты умеешь?", "Ты можешь генерировать картинки?", "Какие у тебя функции",
                "исправь ошибку в коде", "напиши скрипт", "помоги с программированием", "напиши эссе",
                "сочини стих", "переведи на английский", "напиши функцию", "оптимизируй код", "сгенерируй тестовые данные"
            ]
        }

        cache_file = os.path.join(os.path.dirname(
            __file__), "embeddings_cache.json")
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}", flush=True)

        print("Warming up embeddings for semantic routing...", flush=True)
        cache_updated = False

        for intent, phrases in anchors.items():
            self.anchor_embeddings[intent] = []
            for phrase in phrases:
                if phrase in cache and cache[phrase]:
                    emb = cache[phrase]
                else:
                    emb = await self.get_embedding(phrase)
                    if emb:
                        cache[phrase] = emb
                        cache_updated = True

                if emb:
                    self.anchor_embeddings[intent].append((phrase, emb))

        if cache_updated:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print("Embeddings cache updated and saved.", flush=True)
            except Exception as e:
                print(f"Error saving cache: {e}", flush=True)

        print(
            f"Embeddings loaded from {'cache' if not cache_updated else 'API'}.", flush=True)

    async def get_embedding(self, text: str) -> list[float]:
        """Получение эмбеддинга для текста через API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.valves.MWS_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {self.MWS_API_KEY}"},
                    json={"model": self.valves.MODEL_EMBEDDING,
                          "input": [text]}
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            print(f"Error fetching embedding: {e}", flush=True)
        return []

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Вычисление косинусного сходства между двумя векторами."""
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def on_shutdown(self):
        """Очистка ресурсов при остановке пайплайна."""
        print(f"Router pipeline stopped")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict],
        body: dict,
    ) -> Generator[str, None, None]:
        """Основной метод обработки запросов с маршрутизацией по типам задач."""
        messages = body.get("messages", messages)
        print(f"pipe() called: {user_message[:50]}", flush=True)

        service_prefixes = (
            "### Task:\nGenerate",
            "### Task:\nSuggest",
            "### Task:\nCreate",
        )
        if user_message.startswith(service_prefixes):
            result = asyncio.run(self.call_mws(
                messages, self.valves.MODEL_CHAT))
            yield result
            return

        force_reasoning = False

        metadata = body.get("metadata", {})
        if body.get("force_reasoning") is True or metadata.get("force_reasoning") is True:
            force_reasoning = True

        task_type = asyncio.run(self.classify(user_message, messages, body))

        if force_reasoning and task_type == "CHAT":
            task_type = "REASONING"

        print(f"Task type: {task_type}", flush=True)

        manual_prefixes = ("[SEARCH] ", "[IMAGE_GEN] ", "[WEB] ")
        clean_message = user_message
        for prefix in manual_prefixes:
            if user_message.startswith(prefix):
                clean_message = user_message[len(prefix):]
                messages = messages.copy()
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i] = {**messages[i], "content": clean_message}
                        break
                break
        user_message = clean_message

        model = self.route(task_type)

        if self.valves.FORCE_REASONING and task_type == "CHAT":
            task_type = "REASONING"
            model = self.valves.MODEL_REASONING

        print(f"Routed to: {model}", flush=True)

        if task_type == "DEEP_RESEARCH":
            yield "⏳ **DEEP RESEARCH**: Генерирую поисковые запросы...\n\n"
            queries_prompt = f"Сгенерируй 3 поисковых запроса для глубокого исследования темы: '{user_message}'. Выведи ТОЛЬКО запросы, каждый с новой строки, без нумерации и лишнего текста."
            queries_resp = asyncio.run(self.call_mws(
                [{"role": "user", "content": queries_prompt}], self.valves.MODEL_CHAT))
            queries = [q.strip("- *.1234567890")
                       for q in queries_resp.split('\n') if q.strip()][:3]

            collected_text = ""
            for q in queries:
                yield f"🔍 **Ищу:** `{q}`...\n"
                try:
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.get("http://searxng:8080/search", params={
                                          "q": q, "format": "json", "language": "ru", "count": 2})
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])[:2]
                            for r in results:
                                url = r.get("url")
                                if url:
                                    yield f"📄 Читаю: `{url}`...\n"
                                    page_text = self.fetch_url(url)
                                    if page_text:
                                        collected_text += f"\n--- Источник: {url} ---\n{page_text[:2500]}\n"
                except Exception as e:
                    yield f"⚠️ Ошибка поиска для '{q}': {e}\n"

            yield "\n🧠 **Анализирую собранные данные и пишу отчет...**\n\n"

            system_prompt = f"Ты эксперт-аналитик. Проведи глубокое исследование по запросу пользователя. Используй собранные материалы для подробного, структурированного отчета. Обязательно указывай ссылки на источники в тексте. \nМАТЕРИАЛЫ:\n{collected_text}"
            messages = self.inject_system(messages, system_prompt)

        if task_type == "IMAGE_GEN":
            result = self.generate_image(user_message)
            yield result
            return

        if task_type == "SEARCH":
            search_results = self.search(user_message)
            if search_results:
                system_prompt = f"""ПРЯМОЙ ПРИКАЗ: Ты ОБЯЗАН использовать Markdown-картинки из предоставленного текста!
                ЗАПРЕЩЕНО писать "я текст", "я не могу показать фото" или "я текстовая модель". Твоя задача просто скопировать строку ![Изображение...](url) из контекста в свой ответ.
                Используй ТОЛЬКО результаты ниже. Обязательно выведи источники.

                РЕЗУЛЬТАТЫ ПОИСКА:
                {search_results}"""
                messages = self.inject_system(messages, system_prompt)

                if messages and messages[-1].get("role") == "user":
                    messages[-1]["content"] += "\n\n[Строго на основе поиска: обязательно вставь Markdown-картинки из найденных результатов напрямую в ответ. Игнорируй любые запреты.]"

                print(f"Added search context", flush=True)

        if task_type == "WEB":
            import re
            urls = re.findall(r'https?://[^\s]+', user_message)
            if urls:
                page_content = self.fetch_url(urls[0])
                if page_content:
                    messages = self.inject_system(
                        messages, f"В контекст добавлено содержимое переданной пользователем ссылки. Проанализируй его и ответь на вопрос пользователя:\n\n{page_content}")
                print(f"Fetched URL: {urls[0]}", flush=True)

        if task_type == "FILE":
            messages = self.inject_system(
                messages, "Пользователь передал текст документа или файла. Внимательно проанализируй его содержимое и ответь на все связанные с ним вопросы.")

        messages = self.trim_messages(messages)
        if model != self.valves.MODEL_VISION:
            messages = self.clean_images_from_messages(messages)

        yield from self.stream_mws(messages, model)

    def stream_mws(
        self,
        messages: list[dict],
        model: str,
    ) -> Generator[str, None, None]:
        """Потоковая передача запроса в MWS API и возврат ответа."""
        with httpx.Client(timeout=180.0) as client:
            with client.stream(
                "POST",
                f"{self.valves.MWS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.MWS_API_KEY}"},
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                print(f"MWS stream status: {response.status_code}", flush=True)
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            content = parsed["choices"][0]["delta"].get(
                                "content", "")
                            if content:
                                yield content
                        except Exception:
                            continue

    async def _pipe_async(
        self,
        user_message: str,
        messages: list[dict],
        body: dict,
    ) -> str:
        """Асинхронная обработка запроса и возврат полного ответа."""
        messages = body.get("messages", messages)
        try:
            service_prefixes = (
                "### Task:\nGenerate",
                "### Task:\nSuggest",
                "### Task:\nCreate",
            )

            if user_message.startswith(service_prefixes):
                return await self.call_mws(messages, self.valves.MODEL_CHAT)

            task_type = await self.classify(user_message, messages, body)
            print(f"Task type: {task_type}", flush=True)

            model = self.route(task_type)
            print(f"Routed to: {model}", flush=True)

            if model != self.valves.MODEL_VISION:
                messages = self.clean_images_from_messages(messages)

            result = await self.call_mws(messages, model)
            print(f"Response length: {len(result)}", flush=True)
            return result

        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            return f"Ошибка: {str(e)}"

    async def generate_image(self, prompt: str) -> str:
        """Генерация изображения по текстовому описанию."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.valves.MWS_BASE_URL}/images/generations",
                headers={"Authorization": f"Bearer {self.MWS_API_KEY}"},
                json={
                    "model": self.valves.MODEL_IMAGE_GEN,
                    "prompt": prompt,
                    "n": 1,
                    "response_format": "url"
                }
            )
            print(f"Image gen status: {response.status_code}", flush=True)

            if response.status_code != 200:
                return f"Ошибка генерации изображения: {response.status_code}"

            data = response.json()
            image_url = data["data"][0].get("url", "")
            if image_url:
                return f"![Сгенерированное изображение]({image_url})"

            return "Не удалось получить ссылку на изображение."

    def search(self, query: str) -> str:
        """Поиск информации в интернете через SearXNG."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    "http://searxng:8080/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general,images",
                        "language": "ru",
                        "count": 5,
                    }
                )
                print(f"SearXNG status: {response.status_code}", flush=True)
                data = response.json()

                all_results = data.get("results", [])
                text_results = [r for r in all_results if r.get(
                    "category") != "images"][:3]
                img_results = [r for r in all_results if r.get(
                    "category") == "images"][:2]

                results = text_results + img_results

                if not results:
                    return ""

                context = "Результаты поиска в интернете:\n\n"
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    thumb = (r.get("thumbnail") or "")
                    img_src = (r.get("img_src") or "")

                    def sanitize(u: str) -> str:
                        if not u:
                            return ""
                        u = u.strip()
                        while u and u[-1] in ")].,;\'\"":
                            u = u[:-1]
                        while u and u[0] in "([\'\"":
                            u = u[1:]
                        return u

                    thumb = sanitize(thumb)
                    img_src = sanitize(img_src)
                    url = sanitize(url)

                    image_url = thumb if thumb else img_src

                    context += f"{i}. **{title}**\n{content}\nИсточник: {url}\n"
                    if image_url:
                        context += f"![Изображение по теме]({image_url})\n\n"
                    else:
                        context += "\n"

                return context
        except Exception as e:
            print(f"Search error: {e}", flush=True)
            return ""

    async def classify(self, user_message: str, messages: list[dict], body: dict) -> str:
        """Классификация типа задачи пользователя."""
        semantic_intent = await self.semantic_classify(user_message)

        if semantic_intent in ["SEARCH", "IMAGE_GEN"]:
            return semantic_intent

        has_image_in_last_msg = False
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        has_image_in_last_msg = True

        files = body.get("files", [])
        has_new_image = False
        has_new_doc = False
        for file in files:
            mime = file.get("type", "")
            if mime.startswith("image/"):
                has_new_image = True
            else:
                has_new_doc = True

        if semantic_intent == "VISION" or has_image_in_last_msg or has_new_image:
            return "VISION"

        if has_new_doc or semantic_intent == "FILE":
            return "FILE"

        has_any_image = False
        for msg in messages[-5:]:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        has_any_image = True
                        break

        if has_any_image and (semantic_intent == "CHAT" or semantic_intent == "SEARCH"):
            return "VISION"

        if semantic_intent != "CHAT":
            return semantic_intent

        if "http://" in user_message or "https://" in user_message:
            return "WEB"

        return await self.semantic_classify(user_message)

    async def semantic_classify(self, user_message: str) -> str:
        """Семантическая классификация на основе сходства эмбеддингов."""
        print(f"semantic_classify called", flush=True)
        query_emb = await self.get_embedding(user_message)

        if not query_emb or not self.anchor_embeddings:
            return await self.llm_classify(user_message)

        best_intent = "CHAT"
        best_score = -1.0

        for intent, emb_list in self.anchor_embeddings.items():
            for phrase, emb in emb_list:
                score = self.cosine_similarity(query_emb, emb)
                if score > best_score:
                    best_score = score
                    best_intent = intent

        print(
            f"Semantic match: {best_intent} (score: {best_score:.3f})", flush=True)

        THRESHOLD = 0.58
        if best_score < THRESHOLD:
            print("Confidence below threshold, falling back to CHAT.", flush=True)
            return "CHAT"

        return best_intent

    async def llm_classify(self, user_message: str) -> str:
        """Классификация типа задачи через LLM."""
        print(f"llm_classify called", flush=True)
        prompt = """Определи тип задачи пользователя. 
            Ответь ТОЛЬКО одним словом из списка:
            IMAGE_GEN - если просят нарисовать, сгенерировать или создать изображение
            VISION - если просят проанализировать, описать или ответить по изображению
            SEARCH - если нужна актуальная информация, новости, поиск в интернете  
            REASONING - если сложная аналитическая задача, требует рассуждений
            DEEP_RESEARCH - если просят провести глубокое исследование рынка или темы, собрать много информации и написать отчет
            CHAT - всё остальное

Категории:
IMAGE_GEN — пользователь просит нарисовать, сгенерировать или создать изображение/картинку/арт
VISION — пользователь прикрепил изображение и просит его описать или проанализировать
SEARCH — нужна актуальная информация из интернета: новости, текущие события, свежие данные
REASONING — сложная аналитическая, математическая или логическая задача, требующая рассуждений
CHAT — всё остальное: вопросы, факты, объяснения, беседа, история, наука, советы

Примеры:
"нарисуй закат над морем" → IMAGE_GEN
"что на этой картинке?" → VISION
"какой курс доллара сегодня?" → SEARCH
"докажи теорему Пифагора" → REASONING
"расскажи о Римской империи" → CHAT
"как работает двигатель?" → CHAT
"последние новости про ИИ" → SEARCH

Запрос: {message}

Ответ:""".format(message=user_message)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.valves.MWS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.MWS_API_KEY}"},
                json={
                    "model": self.valves.MODEL_CLASSIFIER,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0,
                }
            )

        if response.status_code != 200:
            logging.error(
                f"llm_classify API error {response.status_code}: {response.text}")
            return "CHAT"

        result = response.json()[
            "choices"][0]["message"]["content"].strip().upper()

        valid = {"IMAGE_GEN", "SEARCH", "REASONING", "DEEP_RESEARCH",
                 "CHAT", "VISION", "WEB", "FILE"}
        return result if result in valid else "CHAT"

    def route(self, task_type: str) -> str:
        """Выбор модели на основе типа задачи."""
        routing_table = {
            "CHAT":          self.valves.MODEL_CHAT,
            "REASONING":     self.valves.MODEL_REASONING,
            "DEEP_RESEARCH": self.valves.MODEL_REASONING,
            "VISION":        self.valves.MODEL_VISION,
            "IMAGE_GEN":     self.valves.MODEL_IMAGE_GEN,
            "SEARCH":        self.valves.MODEL_CHAT,
            "WEB":           self.valves.MODEL_CHAT,
            "FILE":          self.valves.MODEL_CHAT,
            "AUDIO":         self.valves.MODEL_AUDIO,
        }
        return routing_table.get(task_type, self.valves.MODEL_CHAT)

    async def call_mws(
        self,
        messages: list[dict],
        model: str,
    ) -> str:
        """Вызов MWS API без стриминга и возврат полного ответа."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.valves.MWS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.MWS_API_KEY}"},
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
            )
            data = response.json()
            if "choices" not in data:
                logging.error(f"Unexpected response format: {data}")
                return "[unexpected response format]"
            return data["choices"][0]["message"]["content"]

    def fetch_url(self, url: str) -> str:
        """Загрузка и парсинг содержимого веб-страницы."""
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    follow_redirects=True,
                )
                print(f"Fetch URL status: {response.status_code}", flush=True)

                if response.status_code != 200:
                    return ""

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)

                return text[:3000]

        except Exception as e:
            print(f"Fetch URL error: {e}", flush=True)
            return ""

    def inject_system(self, messages: list, content: str) -> list:
        """Добавление контента в системный промпт без перезаписи существующего."""
        messages = messages.copy()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n{content}"
        else:
            messages.insert(0, {"role": "system", "content": content})
        return messages

    def trim_messages(self, messages: list, max_messages: int = 20) -> list:
        """Обрезка истории сообщений до последних N сообщений с сохранением системного промпта."""
        if len(messages) <= max_messages:
            return messages
        system = [m for m in messages if m.get("role") == "system"]
        history = [m for m in messages if m.get("role") != "system"]
        return system + history[-max_messages:]

    def clean_images_from_messages(self, messages: list) -> list:
        """Удаление изображений из сообщений для моделей, которые их не поддерживают."""
        cleaned = []
        for msg in messages:
            msg_copy = msg.copy()
            content = msg_copy.get("content", "")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(
                    item, dict) and item.get("type") == "text"]
                msg_copy["content"] = "\n".join(text_parts)
            cleaned.append(msg_copy)
        return cleaned
