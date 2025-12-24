import asyncio
import logging
import random
from asyncio import Task
from typing import TYPE_CHECKING, Any, cast

from slixmpp import JID

if TYPE_CHECKING:
    from src.bot import SmartXMPPBot


class TypingEffectMixin:
    """Миксин для добавления эффекта печати к XMPP клиенту"""

    def __init__(self, *args: Any, **kwargs: Any):
        self.active_sessions: dict[str, Task] = {}
        self.message_ids: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    async def send_message_with_typing(self, text: str, to_jid: JID, speed: float = 0.02) -> None:
        """Отправить сообщение с эффектом печати."""
        session_id = str(to_jid)
        if session_id in self.active_sessions:
            self.active_sessions[session_id].cancel()
        task = asyncio.create_task(self._typing_task(text, to_jid, speed, session_id))
        self.active_sessions[session_id] = task
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _send_cursor(self, to_jid: JID) -> str | None:
        """Отправляет сообщение с курсором, возвращает его ID"""
        bot = cast("SmartXMPPBot", self)
        try:
            msg_id = await bot.send_msg(message="█", to=to_jid, is_encrypt=False)
            return msg_id
        except Exception as e:
            logging.error(f"Не удалось отправить курсор: {e}")
            try:
                msg = bot.make_message(mto=to_jid)
                msg["body"] = "█"
                msg["type"] = "groupchat" if "@conference." in str(to_jid) else "chat"
                msg.send()
                return msg.get("id") or f"cursor_{id(msg)}"
            except Exception:
                return None

    async def _typing_task(self, text: str, to_jid: JID, speed: float, session_id: str) -> None:
        """Асинхронная задача для эффекта печати."""
        try:
            msg_id = await self._send_cursor(to_jid)
            if not msg_id:
                return

            self.message_ids[session_id] = msg_id
            displayed_text = ""
            words = text.split()

            for i, word in enumerate(words):
                if session_id not in self.active_sessions:
                    break

                displayed_text += word + " "
                cursor = "█" if (i % 2) == 0 else "▌"
                display = displayed_text.strip() + cursor

                await self._edit_message(to_jid, msg_id, display)

                base_delay = speed * len(word) * 0.5
                variation = random.choice([0.85, 1.0, 1.15])
                delay = base_delay * variation
                punctuation_multiplier = 1.0
                if any(word.endswith(p) for p in [".", "!", "?"]):
                    punctuation_multiplier = 1.7
                elif any(word.endswith(p) for p in [",", ";", ":"]):
                    punctuation_multiplier = 1.3
                delay *= punctuation_multiplier
                delay = max(0.25, min(delay, 2.0))
                await asyncio.sleep(delay)

            if session_id in self.active_sessions:
                await self._edit_message(to_jid, msg_id, text)

        except Exception as e:
            logging.error(f"Ошибка в эффекте печати: {e}")
        finally:
            self._cleanup_session(session_id)

    async def _edit_message(self, to_jid: JID, msg_id: str, new_body: str) -> None:
        """Редактирует существующее сообщение"""
        try:
            bot = cast("SmartXMPPBot", self)
            await bot.send_msg(
                message=new_body,
                to=to_jid,
                replace_msg_id=msg_id,
                is_encrypt=False,
            )
        except Exception as e:
            logging.warning(f"Не удалось отредактировать сообщение: {e}")

    def _cleanup_session(self, session_id: str) -> None:
        """Очищает сессию печати"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.message_ids:
            del self.message_ids[session_id]

    def stop_typing(self, to_jid: JID) -> bool:
        """Принудительно останавливает печать"""
        session_id = str(to_jid)
        if session_id in self.active_sessions:
            self.active_sessions[session_id].cancel()
            self._cleanup_session(session_id)
            return True
        return False
