import logging
from pathlib import Path
from typing import Any, FrozenSet

from omemo import DeviceInformation
from slixmpp_omemo import XEP_0384, TrustLevel

from src.custom_omemo.storage import StorageImpl


class XEP_0384Impl(XEP_0384):
    default_config = {
        "fallback_message": "This message is OMEMO encrypted.",
        "json_file_path": None,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__storage: StorageImpl

    def plugin_init(self) -> None:
        if not self.json_file_path:
            raise Exception("JSON file path not specified.")

        self.__storage = StorageImpl(Path(self.json_file_path))
        super().plugin_init()

    @property
    def storage(self):
        return self.__storage

    @property
    def _btbv_enabled(self) -> bool:
        return True

    def _devices_blindly_trusted(self, jid, devices):
        logging.info(f"🔐 Автоматически доверяем устройствам для {jid}: {[d.device_id for d in devices]}")
        return devices

    async def _prompt_manual_trust(
        self, manually_trusted: FrozenSet[DeviceInformation], identifier: str | None
    ) -> None:
        logging.info(f"🔐 Ручное доверие устройствам: {[d.device_id for d in manually_trusted]}")
        for device in manually_trusted:
            self.session_manager.set_trust(device.bare_jid, device.device_id, TrustLevel.TRUSTED)
