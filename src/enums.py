from enum import Enum


class PluginTypes(Enum):
    """Код плагина и его обозначение."""

    SERVICE_DISCOVERY = "xep_0030"
    MULTI_USER_CHAT = "xep_0045"
    XMPP_PING = "xep_0199"
    PUB_SUB = "xep_0060"
    CUSTOM_OMEMO_ENCRYPTION = "XEP_0384Impl"
    CHAT_STATES = "xep_0085"
    USER_AVATARS = "xep_0084"
    V_CARD = "xep_0054"


class MessageType(Enum):
    """Тип сообщения."""

    GROUP_CHAT = "groupchat"
    CHAT = "chat"


class PostType(Enum):
    """Тип поста."""

    ALERT = "alert"
    CANCEL = "cancel"
    UNKNOWN = "unknown"
