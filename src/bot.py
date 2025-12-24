import asyncio
import hashlib
import imghdr
import logging
from asyncio import Task
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from xml.etree.ElementTree import Element

from slixmpp import JID, ClientXMPP, Message
from slixmpp.plugins import register_plugin
from slixmpp.plugins.xep_0084.avatar import AvatarMetadataItem
from slixmpp.types import JidStr
from slixmpp_omemo import XEP_0384

from src.custom_omemo.plugin import XEP_0384Impl
from src.enums import MessageType, PluginTypes
from src.mixins import TypingEffectMixin
from src.services import LLMService
from src.settings import settings
from src.utils import check_ollama_health

register_plugin(XEP_0384Impl, name="XEP_0384Impl")

MessageTypeLiteral = Literal["chat", "error", "groupchat", "headline", "normal"]
ChatStatesLiteral = Literal["composing", "active"]


class SmartXMPPBot(TypingEffectMixin, ClientXMPP):
    """Умный XMPP-Бот."""

    def __init__(self, jid: JidStr, password: str, room: str, nick: str) -> None:
        super().__init__(jid, password)
        self.room = JID(room)
        self.nick = nick
        self.llm_service = LLMService()

        self.MAX_HISTORY_LENGTH: int = 10
        self.MAX_RECONNECT_ATTEMPTS: int = 10
        self.MIN_RESPONSE_INTERVAL_SECONDS: int = settings.MIN_RESPONSE_INTERVAL_SECONDS
        self.DEFAULT_CONTEXT: str = "Контекста нет"

        self.reconnect_attempts: int = 0
        self.message_history: list[dict[str, Any]] = []
        self.last_response_time: datetime = datetime.now()
        self.active_sessions: dict[str, Task] = {}
        self.message_ids: dict[str, str] = {}

        for plugin in [
            PluginTypes.SERVICE_DISCOVERY,
            PluginTypes.MULTI_USER_CHAT,
            PluginTypes.XMPP_PING,
            PluginTypes.PUB_SUB,
            PluginTypes.CHAT_STATES,
            PluginTypes.USER_AVATARS,
            PluginTypes.V_CARD,
        ]:
            self.register_plugin(plugin.value)
        self.register_plugin(
            PluginTypes.CUSTOM_OMEMO_ENCRYPTION.value,
            {"json_file_path": "omemo_data.json"},
            module=__name__,
        )

        self.add_event_handler("session_start", self.initialize)
        self.add_event_handler("got_online", self.join_muc_room)
        self.add_event_handler("groupchat_message", self.muc_message)
        self.add_event_handler("session_end", self.handle_disconnect)
        self.add_event_handler("disconnected", self.handle_disconnect)

    async def initialize(self, event: Any = None) -> None:
        """Инициализация бота."""
        await self.get_roster()
        self.send_presence()
        await self.set_avatar(image_path="./static/avatar.jpg")
        welcome_message = "AI-Бот запущен и готов к работе"
        logging.info(welcome_message)
        await self.send_message_admin(message=f"🤖 {welcome_message}!")

    async def handle_disconnect(self, event: Any = None) -> None:
        """Обработка отключения от сервера."""
        logging.warning("Соединение с сервером потеряно")
        if self.reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            self.reconnect_attempts += 1
            wait_time = min(2**self.reconnect_attempts, 60)
            logging.info(
                f"Попытка переподключения {self.reconnect_attempts}/"
                f"{self.MAX_RECONNECT_ATTEMPTS} через {wait_time} сек..."
            )
            await asyncio.sleep(wait_time)
            self.reconnect()
        else:
            logging.error("Превышено максимальное количество попыток переподключения")

    async def set_avatar(self, image_path: str) -> None:
        """Установить аватар для бота."""
        try:
            if not Path(image_path).exists():
                logging.error(f"Файл аватара не найден: {image_path}")
                return None

            with open(image_path, "rb") as f:
                image_data = f.read()

            avatar_hash = hashlib.sha1(image_data).hexdigest()
            image_type = imghdr.what(None, h=image_data)
            mime_type = f"image/{image_type}" if image_type else "image/jpeg"
            await self.plugin[PluginTypes.USER_AVATARS.value].publish_avatar(  # type: ignore[typeddict-item]
                data=image_data,
            )
            metadata_items = AvatarMetadataItem(id=avatar_hash, type=mime_type, bytes=len(image_data))
            await self.plugin[
                PluginTypes.USER_AVATARS.value  # type: ignore[typeddict-item]
            ].publish_avatar_metadata(items=metadata_items)
        except Exception as e:
            logging.error(f"Произошла ошибка при установке аватара: {e}")

    async def join_muc_room(self, event: Any = None) -> None:
        """Подключиться к группе."""
        logging.info("Бот online, присоединяюсь к комнате...")
        try:
            await self.plugin[PluginTypes.MULTI_USER_CHAT.value].join_muc(  # type: ignore[typeddict-item]
                room=self.room,
                nick=self.nick,
            )
            logging.info(f"Бот присоединился к комнате: {self.room} как {self.nick}")
        except Exception as e:
            logging.error(f"Ошибка присоединения к комнате: {e}")

    async def send_message_admin(self, message: str) -> None:
        """Отправить сообщение администратору."""
        if admin_jid := JID(settings.ADMIN_JID):
            try:
                await self.send_msg(to=admin_jid, message=message, message_type="chat")
                logging.info("Уведомление отправлено администратору")
            except Exception as e:
                logging.error(f"❌ Ошибка уведомления администратора: {e}")

    async def send_chat_state(self, state: ChatStatesLiteral):
        """Отправить уведомление о наборе сообщения."""
        try:
            msg = self.Message()
            msg["to"] = self.room
            msg["type"] = "groupchat"
            msg["id"] = self.new_id()
            msg["chat_state"] = state
            msg.send()
        except Exception as e:
            logging.error(f"Произошла ошибка при отправке уведомления о наборе сообщения: {e}")

    async def send_msg(
        self,
        *,
        message: str,
        to: JID | str | None = None,
        message_type: MessageTypeLiteral = "groupchat",
        is_encrypt: bool = True,
        is_mentions: bool = False,
        replace_msg_id: str | None = None,
    ) -> str | None:
        """Отправить сообщение."""
        if not to:
            to = self.room
        if isinstance(to, str):
            to = JID(to)

        if is_mentions and message_type == MessageType.GROUP_CHAT.value:
            room_users = self.plugin[
                PluginTypes.MULTI_USER_CHAT.value  # type: ignore[typeddict-item]
            ].get_roster(self.room)
            if room_users:
                mention_text = ", ".join([f"{nick}" for nick in room_users if nick != self.nick])
                message = f"{mention_text}\n{message}"

        if is_encrypt:
            msg_id = await self._encrypt_and_send_message(
                message=message,
                to=to,
                message_type=message_type,
                replace_msg_id=replace_msg_id,
            )
        else:
            try:
                msg = self.make_message(mto=to, mbody=message, mtype=message_type)
                if replace_msg_id:
                    self._add_replace_elem(msg, replace_msg_id)
                msg.send()
                msg_id = msg.get("id")
            except Exception as e:
                logging.error(f"❌ Ошибка отправки без шифрования: {e}")
                msg_id = None
        logging.info(f"{'Зашифрованный' if is_encrypt else 'Обычный'} ответ отправлен в {message_type}")
        return msg_id

    async def send_debug_message(self, message: str, is_reply_admin: bool = False) -> None:
        """Отправить debug-сообщение."""
        if settings.IS_DEBUG:
            await self.send_msg(message=f"❗️❗❗ DEBUG ❗❗❗ \n\n {message}")
            if is_reply_admin:
                await self.send_message_admin(message=f"❗️❗❗ DEBUG ❗❗❗ \n\n {message}")

    async def muc_message(self, msg: Message) -> None:
        """Обработать полученное сообщение."""
        mtype = msg["type"]
        if mtype not in {"chat", "normal", "groupchat"}:
            return None

        if msg["mucnick"] == self.nick:
            return None

        try:
            omemo_plugin: XEP_0384 = self.plugin[
                PluginTypes.CUSTOM_OMEMO_ENCRYPTION.value  # type: ignore[typeddict-item]
            ]
            if omemo_plugin.is_encrypted(msg):
                message, device_info = await omemo_plugin.decrypt_message(msg)
                body = message.get("body", "") if message else ""
                logging.debug(f"Сообщение расшифровано от {msg['mucnick']}")
            else:
                body = msg["body"]
                if "OMEMO" in body and "doesn't support" in body:
                    return None
                logging.debug(f"Обычное сообщение от {msg['mucnick']}")

            if not body:
                return None

            self._add_to_history(body=body, sender=msg["mucnick"])

            if self._too_soon_to_respond():
                too_soon_message = "Слишком рано после предыдущего ответа, пропускаю"
                await self.send_debug_message(message=too_soon_message)
                logging.info(too_soon_message)
                return None

            if not await check_ollama_health():
                await self.send_debug_message(message="Ollama не подключена", is_reply_admin=True)
                return None

            should_respond, reason = await self.llm_service.analyze_conversation(self.message_history)
            logging.debug(f"Решение анализа: {should_respond} - {reason}")

            if not should_respond:
                reason_message = f"Пропускаю. Причина:\n\n{reason}"
                await self.send_debug_message(message=reason_message)
                logging.debug(reason_message)
                return None

            await self.send_debug_message(message=f"Причина ответа:\n\n{reason}")

            try:
                await self.send_chat_state(state="composing")
                context = await self.llm_service.analyze_context(self.message_history)

                if not context:
                    logging.error("Контекста нет.")
                    context = self.DEFAULT_CONTEXT

                await self.send_debug_message(message=f"Контекст беседы:\n\n{context}")
                code = await self.llm_service.detector_code(self.message_history)
                await self.send_debug_message(
                    message=f"Детектор кода:\n\n{code}",
                    is_reply_admin=True,
                )
                if code and code.get("is_programming"):
                    response = await self.llm_service.generate_code_response(self.message_history)
                else:
                    response = await self.llm_service.generate_response(  # type: ignore[assignment]
                        conversation_history=self.message_history, context=context or self.DEFAULT_CONTEXT
                    )
                if response:
                    self._add_to_history(body=response, sender=self.nick)
                    logging.info("Ответ сгенерирован! Отправляю...")
                    await self.send_chat_state(state="active")
                    if settings.ENABLE_TYPING_EFFECT:
                        await self.send_message_with_typing(text=response, to_jid=self.room)
                    else:
                        await self.send_msg(message=response)
                    self.last_response_time = datetime.now()
                else:
                    logging.warning("LLM не сгенерировал ответ")
            except Exception as e:
                logging.error(f"Ошибка генерации ответа: {e}")
                await self.send_chat_state(state="active")
                self.stop_typing(self.room)
                await self.send_message_admin(message=f"Ошибка: {str(e)[:50]}")

        except Exception as e:
            logging.error(f"Общая ошибка обработки: {e}")
            await self.send_chat_state(state="active")
            self.stop_typing(self.room)
            raise e

    async def _encrypt_and_send_message(
        self,
        message: str,
        to: JID | set[JID],
        message_type: MessageTypeLiteral = "groupchat",
        replace_msg_id: str | None = None,
    ) -> str | None:
        """Зашифровать и отправить сообщение."""
        try:
            omemo_plugin = self.plugin[
                PluginTypes.CUSTOM_OMEMO_ENCRYPTION.value  # type: ignore[typeddict-item]
            ]
            reply_msg = self.make_message(mto=to, mbody=message, mtype=message_type)  # type: ignore[arg-type]

            if replace_msg_id:
                self._add_replace_elem(reply_msg, replace_msg_id)

            if message_type == MessageType.GROUP_CHAT.value:
                to = self.get_encrypt_for_muc()
            messages, errors = await omemo_plugin.encrypt_message(reply_msg, to)
            msg_id = None
            for _, encrypted_msg in messages.items():
                encrypted_msg.send()
                if not msg_id and encrypted_msg.get("id"):
                    msg_id = encrypted_msg["id"]
            if not msg_id:
                msg_id = f"msg_{int(asyncio.get_event_loop().time() * 1000)}"
            return msg_id
        except Exception as e:
            logging.error(f"Ошибка шифрования: {e}")
            fallback_msg = self.make_message(
                mto=to, mbody=message, mtype=message_type  # type: ignore[arg-type]
            )
            if replace_msg_id:
                self._add_replace_elem(fallback_msg, replace_msg_id)
            fallback_msg.send()
            return fallback_msg.get("id")

    def get_encrypt_for_muc(self) -> set[JID]:
        """Получить множество JID, для кого нужно шифровать сообщение."""
        xep_0045 = self.plugin[PluginTypes.MULTI_USER_CHAT.value]  # type: ignore[typeddict-item]
        encrypt_for: set[JID] = set()
        for nick in xep_0045.get_roster(self.room):
            if nick.lower() != self.nick.lower() and (
                jid_property := xep_0045.get_jid_property(self.room, nick, "jid")
            ):
                encrypt_for.add(JID(jid_property))
        return encrypt_for

    @staticmethod
    def _add_replace_elem(message: Message, replace_msg_id: str) -> None:
        """Добавить пометку об изменении сообщения."""
        replace_ns = "urn:xmpp:message-correct:0"
        replace_elem = Element(f"{{{replace_ns}}}replace")
        replace_elem.set("id", replace_msg_id)
        message.xml.append(replace_elem)

    def _add_to_history(self, body: str, sender: str) -> None:
        """Добавляет сообщение в историю."""
        self.message_history.append(
            {
                "sender": sender,
                "text": body.replace(self.nick, ""),
                "time": datetime.now().strftime(format="%m-%d-%Y %H:%M:%S"),
            }
        )

        if len(self.message_history) > self.MAX_HISTORY_LENGTH:
            self.message_history.pop(0)

    def _too_soon_to_respond(self) -> bool:
        """Проверяет, не слишком ли рано для нового ответа."""
        elapsed = datetime.now() - self.last_response_time
        return elapsed.total_seconds() < self.MIN_RESPONSE_INTERVAL_SECONDS
