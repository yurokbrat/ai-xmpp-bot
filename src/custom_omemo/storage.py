import json
from pathlib import Path

from omemo import JSONType
from omemo.storage import Just, Maybe, Nothing, Storage


class StorageImpl(Storage):
    def __init__(self, json_file_path: Path) -> None:
        super().__init__()
        self.__json_file_path = json_file_path
        self.__data: dict[str, JSONType] = {}
        try:
            with open(self.__json_file_path, encoding="utf8") as f:
                self.__data = json.load(f)
        except Exception:
            pass

    async def _load(self, key: str) -> Maybe[JSONType]:
        if key in self.__data:
            return Just(self.__data[key])
        return Nothing()

    async def _store(self, key: str, value: JSONType) -> None:
        self.__data[key] = value
        with open(self.__json_file_path, "w", encoding="utf8") as f:
            json.dump(self.__data, f)

    async def _delete(self, key: str) -> None:
        self.__data.pop(key, None)
        with open(self.__json_file_path, "w", encoding="utf8") as f:
            json.dump(self.__data, f)
