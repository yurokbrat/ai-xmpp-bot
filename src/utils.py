import logging

import aiohttp

from src.settings import settings


async def check_ollama_health():
    """Проверяет доступность Ollama API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5) as response:
                if response.status == 200:
                    logging.info("Ollama запущен и доступен")
                    return True
                else:
                    logging.error(f"Ollama вернул статус {response.status}")
                    return False
    except Exception as e:
        logging.error(f"❌ Не удалось подключиться к Ollama: {e}")
        return False
