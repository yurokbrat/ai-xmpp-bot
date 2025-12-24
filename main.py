import asyncio
import logging

from src.bot import SmartXMPPBot
from src.settings import settings
from src.utils import check_ollama_health


async def main():
    logging.basicConfig(
        level=settings.LOGGING_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not await check_ollama_health():
        logging.error("Ollama не подключена. AI-агент не работает.")
    bot = SmartXMPPBot(settings.BOT_JID, settings.BOT_PASSWORD, settings.MUC_ROOM, settings.BOT_NICK)
    await bot.connect()
    await asyncio.sleep(10)
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
