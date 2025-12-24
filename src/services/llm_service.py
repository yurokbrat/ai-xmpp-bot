import asyncio
import logging
import random
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

from src.settings import settings


class LLMService:
    """Сервис для работы с LLM-моделями."""

    def __init__(self):
        self.llm_general = OllamaLLM(
            model=settings.AI_DEFAULT_MODEL,
            temperature=0.1,
        )
        self.llm_code = OllamaLLM(
            model=settings.AI_CODE_MODEL,
            temperature=0.1,
        )
        self.decision_chain = self._create_decision_chain()
        self.context_chain = self._create_context_chain()
        self.response_chain = self._create_response_chain()

        self.code_detector_chain = self._create_code_detector_chain()
        self.code_response_chain = self._create_code_response_chain()

        self.is_generating = False
        self.generation_lock = asyncio.Lock()

    def _create_decision_chain(self):
        """Создать цепочку для принятия решения отвечать или нет."""
        prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""{conversation}

        You are an participant in this chat.

        Should you say something?

        Answer YES if:

        DEFINITELY YES:
        - Someone mentioned bot/AI/AI-бот
        - There's a question with "?"
        - Someone asks for help
        - {should_intervene} (chance to join conversation)
        - Can add useful information

        UP TO YOU:
        - Any interesting topic
        - Feel like chatting

        NO only if:
        - Very personal conversation
        - Just replied recently
        - A short phrase not addressed to you

        IMPORTANT: If words "AI-бот", "Бот", "ИИ", "AI" appear in text - almost always YES!
        Be sociable! Participate when possible.

        Answer ONLY in format: DECISION | REASON
            DECISION: YES or NO
            REASON: brief reason (3-5 words)

        Examples answer:
            YES | Contacted the bot
            YES | Have a question
            NO | Just joking
        """,
        )
        return (
            {
                "conversation": RunnablePassthrough(),
                "should_intervene": lambda x: "Consider joining"
                if random.random() < 0.3
                else "No special reason",
            }
            | prompt
            | self.llm_general
            | StrOutputParser()
        )

    def _create_context_chain(self):
        """Создать цепочку для формирования контекста чата."""

        prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""Message: "{conversation}"

            Analyze and respond in Russian BUT use this exact format:
            Topic=[code/greeting/work/humor/other]
            Type=[question/statement/joke]
            Mood=[friendly/neutral/funny/serious/angry]
            Theme= Write the topic of the conversation in a few words

            DO NOT ADD ANY OTHER TEXT!

            Example output:
            Topic=greeting Type=question Mood=friendly Theme=Food Choices
            Topic=code Type=question Mood=serious Theme=Python decorators
            """,
        )
        return (
            {
                "conversation": RunnablePassthrough(),
            }
            | prompt
            | self.llm_general
            | StrOutputParser()
        )

    def _create_code_detector_chain(self):
        """Определяет, связан ли вопрос с программированием"""
        prompt = PromptTemplate(
            input_variables=["message"],
            template="""Analyze if this Russian message is about programming/coding/IT/DataBase/Git:

        Message: "{message}"

        Answer with JSON ONLY:
        {{
          "is_programming": true/false,
          "confidence": 0.0-1.0,
        }}

        Examples:
        Message: "Что такое классы в Python?"
        Response: {{"is_programming": true, "confidence": 0.95}}

        Message: "Какой сегодня день?"
        Response: {{"is_programming": false, "confidence": 0.99}}

        Message: "Почему не работает мой код?"
        Response: {{"is_programming": true, "confidence": 0.98}}

        Your JSON response:""",
        )
        return {"message": RunnablePassthrough()} | prompt | self.llm_general | JsonOutputParser()

    def _create_code_response_chain(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Ты эксперт по программированию. Отвечай НА РУССКОМ.

            Вопрос: {question}

            Правила ответа:
            1. Отвечай ТОЛЬКО на русском языке
            2. Объясни кратко (2-3 предложения)
            3. Если нужен пример кода - приведи короткий
            4. Не повторяй вопрос

            Твой ответ на русском:""",
        )
        return {"question": RunnablePassthrough()} | prompt | self.llm_code | StrOutputParser()

    def _create_response_chain(self):
        """Создать цепочку для формирования ответа."""

        prompt = PromptTemplate(
            input_variables=["conversation", "context_analysis"],
            template="""Ты - дружелюбный помощник в чате. Твоё имя - AI-бот.
            С тобой общаются как с живым собеседником.

            КОНТЕКСТ БЕСЕДЫ:
            {context_analysis}

            ПОСЛЕДНИЕ СООБЩЕНИЯ:
            {conversation}

            ПРАВИЛА ОТВЕТА:
            1. Отвечай ЕСТЕСТВЕННО, как человек в беседе
            2. Отвечай Средними предложениями (3-4 предложения)
            3. Соответствуй настроению и стилю общения чата
            4. Полностью поддерживай атмосферу и контекст беседы.
            5. Не подписывайся и не добавляй префиксы!

            ВАЖНО: Не пытайся ответить на все сообщения сразу! Только на последнее.

            Всегда давай понятный и чёткий ответ!

            Твой ответ строго на русском языке сплошным текстом:""",
        )
        return (
            {
                "conversation": RunnablePassthrough(),
                "context_analysis": RunnablePassthrough(),
            }
            | prompt
            | self.llm_general
            | StrOutputParser()
        )

    @staticmethod
    def _format_conversation(messages: list[dict[str, Any]]) -> str:
        """Форматирует список сообщений в текст"""
        if not messages:
            return "История пуста"

        formatted = []
        for i, msg in enumerate(messages, 1):
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            time = msg.get("time", "")
            formatted.append(f"{i}. {time} - {sender}: {text}")

        return "\n".join(formatted)

    async def analyze_conversation(
        self, conversation_history: list[dict[str, Any]]
    ) -> tuple[bool | None, str]:
        """Анализирует, нужно ли отвечать"""
        if self.is_generating:
            logging.warning("Пропускаю запрос: уже идет генерация ответа")
            return None, "Пропускаю запрос: уже идет генерация ответа"
        conv_text = self._format_conversation(conversation_history[-5:])
        self.is_generating = True
        logging.info(f"Анализ истории чата ({len(conversation_history)} сообщений)...")
        decision_result = await self.decision_chain.ainvoke(conv_text)
        self.is_generating = False
        try:
            if "|" in decision_result:
                parts = decision_result.split("|", 2)
                decision = parts[0].strip().upper()
                reason = parts[1].strip() if len(parts) > 1 else "Нет причины"
                if decision in ["YES", "ДА", "Y"]:
                    return True, reason
                else:
                    return False, reason
            else:
                result_upper = decision_result.upper()
                if any(word in result_upper for word in ["YES", "ДА", "Y"]):
                    return True, "Автоматический анализ (YES найдено в тексте)"
                else:
                    return False, "Автоматический анализ (NO по умолчанию)"
        except Exception as e:
            logging.error(f"Ошибка парсинга решения: {e}, raw: {decision_result}")
            return False, f"Ошибка парсинга: {str(e)[:50]}"

    async def analyze_context(self, conversation_history: list[dict[str, Any]]) -> str | None:
        if self.is_generating:
            logging.warning("Пропускаю запрос: уже идет анализ контекста")
            return None
        conv_text = self._format_conversation(conversation_history[-3:])
        logging.info("Анализ контекста...")
        context_result = await self.context_chain.ainvoke(
            {
                "conversation": conv_text,
            }
        )
        logging.debug(f"Контекст LLM: {context_result}")
        return context_result

    async def detector_code(self, conversation_history: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.is_generating:
            logging.warning("Пропускаю запрос: уже идет анализ контекста")
            return None
        conv_text = self._format_conversation(conversation_history[-1:])
        logging.info("Анализ контекста на код...")
        context_result = await self.code_detector_chain.ainvoke(
            {
                "conversation": conv_text,
            }
        )
        logging.debug(f"Code Detector from LLM: {context_result}")
        return context_result

    async def generate_code_response(self, conversation_history: list[dict[str, Any]]) -> str:
        """Генерирует ответ на вопрос о коде"""

        logging.debug(f"Использование {settings.AI_CODE_MODEL} для генерации ответа...")
        last_message = conversation_history[-1]["text"]
        response = await self.code_response_chain.ainvoke(last_message)
        return response

    async def generate_response(
        self, conversation_history: list[dict[str, Any]], context: str | None
    ) -> str | None:
        """Генерирует ответ после положительного анализа."""

        if self.is_generating:
            logging.warning("Пропускаю запрос: уже идет генерация ответа")
            return None

        async with self.generation_lock:
            try:
                logging.debug(f"Использование {settings.AI_DEFAULT_MODEL} для генерации ответа...")
                self.is_generating = True
                conv_text = self._format_conversation(conversation_history[-3:])
                response = await self.response_chain.ainvoke(
                    {"conversation": conv_text, "context_analysis": context}
                )
                return response
            finally:
                self.is_generating = False
                logging.info("Генерация ответа завершена")
