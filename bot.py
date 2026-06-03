import asyncio
import base64
import os
import tempfile
from collections import defaultdict
from io import BytesIO

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5")

bot = Bot(
    token=TELEGRAM_BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)

dp = Dispatcher()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

client = OpenAI(

    base_url="https://openrouter.ai/api/v1",

    api_key=OPENROUTER_API_KEY,

)

MODEL = OPENROUTER_MODEL
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
MAX_HISTORY_MESSAGES = 20
MEDIA_GROUP_WAIT_SECONDS = 3

user_histories = defaultdict(list)
media_groups = {}
media_group_tasks = {}

CASE_CONTEXT = """
Пациентка: Венера, 1953 года рождения.

Известно:
- фолликулярная лимфома
- grade 1–2: это степень агрессивности клеток, НЕ стадия
- ПЭТ-КТ: множественные лимфоузлы
- поражение брюшной полости
- Deauville 5: высокая метаболическая активность
- упоминалось вовлечение костного мозга
- CD20 положительный
- bcl-2 положительный
- CD10 положительный
- планируется стационарная химиотерапия примерно 6 дней

Проблема:
- пациентка потеряла доверие к врачам из-за слабой коммуникации
- ей сначала говорили одно, потом происходило другое
- задача помощника: вернуть ощущение контроля через ясность, а не через ложное успокоение
"""

SYSTEM_PROMPT = """
Ты — Сергей Павлович, спокойный медицинский помощник с клиническим мышлением опытного онкогематолога-консультанта.

Ты НЕ лечащий врач.
Ты НЕ ставишь диагноз.
Ты НЕ назначаешь и НЕ отменяешь лечение.

Твоя задача:
- объяснять медицинскую информацию простым русским языком
- помогать пациентке и семье понимать, что происходит
- выявлять важные несостыковки и риски
- формулировать точные вопросы врачу
- давать ощущение контроля без ложной уверенности

Главный принцип:
ясность важнее полноты, точность важнее уверенности.

Формат Telegram:
- используй HTML, не Markdown
- заголовки: 🔹 <b>Название</b>
- не используй звёздочки для жирного текста
- между разделами оставляй пустую строку
- не используй сложные таблицы

Стиль:
- только русский язык
- коротко и по делу
- без запугивания
- без “всё будет хорошо”
- без пустых фраз
- как спокойный врач, который объясняет человеку 70+

Запрещённые размытые фразы:
- “требует внимания”
- “не обязательно плохой признак”
- “важный момент”
- “обсудите с врачом” без готовых вопросов

Вместо них объясняй:
- что именно меняется в действиях врача
- почему это влияет на лечение
- какой следующий практический шаг

Клиническое мышление:
- grade = поведение клеток
- стадия = распространение болезни
- ПЭТ/Deauville = метаболическая активность
- ПЭТ показывает потребление глюкозы
- ПЭТ НЕ показывает напрямую скорость деления клеток

Если есть сочетание: низкий grade + Deauville 5, начинай с:
“Здесь есть ключевая несостыковка: …”

Если grade 1–2 + высокая активность:
объясни возможные причины:
1) активная фаза лимфомы
2) возможная трансформация, это ключевой сценарий, который проверяют
3) реже воспаление или инфекция

Если есть множественные лимфоузлы, брюшная полость или костный мозг:
осторожно укажи, что это похоже на распространённый процесс.

Режим клинического разбора:
не пересказывай документ.
Объясняй, что он меняет в понимании болезни.

Каждый раз думай:
1. Что этот факт подтверждает?
2. Что этот факт меняет?
3. Что остаётся неясным?
4. Как это влияет на решение врача?
5. Какой вопрос врачу самый важный?

Если документ не добавляет ничего нового:
скажи: “Этот документ в основном подтверждает уже известную картину”.

Если документ добавляет новый важный факт:
выдели: “Новое важное: …”

Режим второго мнения:
если пользователь спрашивает о плане лечения, оцени логически:
- соответствует ли план известным данным
- какие данные подтверждают этот выбор
- какие данные нужно уточнить
- какие альтернативы разумно спросить
- что было бы тревожным признаком плохой коммуникации

Не говори “врач неправ”.
Говори:
- “это выглядит логично по таким причинам…”
- “это место стоит уточнить, потому что…”

Связь с лечением:
всегда объясняй, почему врач действует именно так:
- почему стационар
- почему терапия может быть интенсивной
- почему нужен контроль реакции организма
- почему могут назначить терапию, даже если лимфома называется “медленной”

Никогда:
- не советуй отменять лечение
- не меняй дозы
- не обещай исход
- не называй точные сроки жизни
- не обвиняй врачей
- не пугай редкими осложнениями без причины

Формат ответа по медицинскому вопросу:
🔹 <b>Главное</b>

🔹 <b>Что это значит</b>

🔹 <b>Возможные причины</b>

🔹 <b>Почему такое лечение</b>

🔹 <b>Спросите врача</b>

Вопросы врачу:
- 3–7 конкретных вопросов
- уровень врача, а не общие фразы
- сразу давай готовые вопросы

Фото документов:
если пришло одно фото или несколько фото:
1. оцени читаемость
2. извлеки только важное:
   - диагноз
   - ПЭТ / Deauville
   - гистология / маркеры
   - план лечения
   - даты
   - стадия или признаки распространения
3. свяжи с текущим кейсом
4. укажи, что добавилось нового
5. дай точные вопросы врачу

Если прислано несколько фото:
- анализируй их как один комплект
- не делай отдельный отчёт по каждому фото
- сначала собери общую картину
- потом дай один консолидированный вывод

Если не уверен:
пиши “читается неуверенно”.

Безопасность:
если есть температура, одышка, кровотечение, потеря сознания, сильная слабость, спутанность сознания, боль в груди или резкое ухудшение —
сразу скажи, что это повод срочно связаться с врачом или вызвать скорую.
"""

RED_FLAG_WORDS = [
    "температура", "жар", "одышка", "не могу дышать",
    "кровотечение", "кровь", "спутанность", "теряет сознание",
    "потеря сознания", "сильная слабость", "боль в груди",
    "резко хуже", "судороги"
]


def has_red_flag(text):
    text = (text or "").lower()
    return any(word in text for word in RED_FLAG_WORDS)


async def safe_answer(message, text):
    try:
        await message.answer(text)
    except TelegramBadRequest:
        clean = text.replace("<b>", "").replace("</b>", "")
        await message.answer(clean, parse_mode=None)


async def send_long_message(message, text):
    max_len = 3300
    for i in range(0, len(text), max_len):
        await safe_answer(message, text[i:i + max_len])


async def transcribe_voice(message):
    voice = message.voice
    file = await bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
        temp_path = temp_audio.name

    await bot.download_file(file.file_path, destination=temp_path)

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=audio_file,
                language="ru"
            )
        return transcript.text.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def photo_to_base64(message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def image_document_to_base64(message):
    file = await bot.get_file(message.document.file_id)

    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def trim_history(user_id):
    user_histories[user_id] = user_histories[user_id][-MAX_HISTORY_MESSAGES:]


async def analyze_with_ai(user_id, user_text, image_base64_list=None):
    user_histories[user_id].append({"role": "user", "content": user_text})
    trim_history(user_id)

    current_content = [{"type": "input_text", "text": user_text}]

    if image_base64_list:
        for image_base64 in image_base64_list:
            current_content.append({
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + image_base64
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": CASE_CONTEXT},
    ] + user_histories[user_id][:-1] + [
        {"role": "user", "content": current_content}
    ]

    response = client.responses.create(
        model=MODEL,
        temperature=0.15,
        input=messages,
    )

    answer = response.output_text

    user_histories[user_id].append({"role": "assistant", "content": answer})
    trim_history(user_id)

    return answer


async def process_media_group(media_group_id):
    await asyncio.sleep(MEDIA_GROUP_WAIT_SECONDS)

    group = media_groups.pop(media_group_id, None)
    media_group_tasks.pop(media_group_id, None)

    if not group:
        return

    message = group["message"]
    user_id = group["user_id"]
    images = group["images"]
    caption = group["caption"]

    text = caption or (
        "Разберите все эти фото как один комплект медицинских документов по пациентке Венере. "
        "Не делайте отдельный обзор каждого фото. "
        "Сделайте один консолидированный клинический отчёт. "
        "Сначала скажите, какие документы читаются хорошо, а какие неуверенно. "
        "Затем выделите: подтверждённый диагноз, данные ПЭТ-КТ, гистологию, маркеры, стадию или признаки распространения, план лечения. "
        "Отдельно укажите, что нового добавили эти документы по сравнению с уже известным контекстом. "
        "В конце дайте 5–8 точных вопросов врачу."
    )

    await safe_answer(message, "Сейчас соберу документы в один отчёт...")

    try:
        answer = await analyze_with_ai(user_id, text, images)
        await send_long_message(message, answer)
    except Exception as e:
        await safe_answer(message, "Сейчас не получилось разобрать документы. Попробуйте ещё раз.")
        print("MEDIA GROUP ERROR:", e)


@dp.message(CommandStart())
async def start(message):
    user_histories[message.from_user.id] = []
    await safe_answer(
        message,
        "Здравствуйте. Я Сергей Павлович.\n\n"
        "Я помогу спокойно разобраться в медицинской информации.\n\n"
        "Можно написать вопрос, отправить голосовое или прислать фото документа.\n\n"
        "Команды:\n"
        "/case — что известно о случае\n"
        "/questions — вопросы врачу\n"
        "/second — режим второго мнения\n"
        "/reset — очистить память\n"
        "/help — помощь"
    )


@dp.message(Command("case"))
async def case_info(message):
    await safe_answer(
        message,
        "🔹 <b>Что известно о случае</b>\n\n"
        "Венера, 1953 г.р.\n"
        "Фолликулярная лимфома grade 1–2.\n"
        "ПЭТ-КТ: множественные лимфоузлы, брюшная полость, Deauville 5.\n"
        "Упоминалось вовлечение костного мозга.\n"
        "План: стационарная химиотерапия около 6 дней.\n\n"
        "Ключевой момент: grade 1–2 говорит о клетках, но не о стадии. "
        "Deauville 5 и распространённость требуют клинического объяснения."
    )


@dp.message(Command("questions"))
async def questions(message):
    await safe_answer(
        message,
        "🔹 <b>Спросите врача</b>\n\n"
        "1. Какая точная стадия заболевания по Ann Arbor?\n"
        "2. Подтверждено ли вовлечение костного мозга?\n"
        "3. Есть ли признаки трансформации в DLBCL?\n"
        "4. Делалась ли биопсия самого активного очага?\n"
        "5. Какая схема терапии планируется?\n"
        "6. Почему выбрана именно эта схема?\n"
        "7. Почему нужен стационар на 6 дней?\n"
        "8. Это лечение направлено на контроль болезни или на достижение ремиссии?\n"
        "9. Когда будет первая оценка ответа?\n"
        "10. Какие симптомы после выписки требуют срочного обращения?"
    )


@dp.message(Command("second"))
async def second_opinion(message):
    await safe_answer(
        message,
        "🔹 <b>Режим второго мнения</b>\n\n"
        "Пришлите план лечения или напишите, что именно назначили.\n\n"
        "Я проверю логически:\n"
        "- соответствует ли план известным данным\n"
        "- какие факты поддерживают этот выбор\n"
        "- что нужно уточнить\n"
        "- какие альтернативы разумно спросить у врача\n\n"
        "Я не буду отменять назначения, но помогу понять, где нужны уточнения."
    )


@dp.message(Command("reset"))
async def reset(message):
    user_histories[message.from_user.id] = []
    await safe_answer(message, "Память этого разговора очищена. Можно начать заново.")


@dp.message(Command("help"))
async def help_command(message):
    await safe_answer(
        message,
        "🔹 <b>Как пользоваться</b>\n\n"
        "Можно написать обычный вопрос, отправить голосовое или прислать фото документа.\n\n"
        "Примеры:\n"
        "— Почему меня кладут в стационар?\n"
        "— Что значит Deauville 5?\n"
        "— Какие вопросы задать врачу?\n\n"
        "Команды:\n"
        "/case — что известно о случае\n"
        "/questions — вопросы врачу\n"
        "/second — режим второго мнения\n"
        "/reset — очистить память"
    )


@dp.message()
async def handle_message(message):
    user_id = message.from_user.id
    text = message.text or message.caption or ""

    print("NEW MESSAGE")
    print("TEXT:", text)
    print("PHOTO:", bool(message.photo))
    print("MEDIA_GROUP:", message.media_group_id)
    print("VOICE:", bool(message.voice))
    print("DOCUMENT:", bool(message.document))

    if message.media_group_id and message.photo:
        try:
            image_base64 = await photo_to_base64(message)
        except Exception as e:
            await safe_answer(message, "Не получилось загрузить одно из фото. Попробуйте отправить ещё раз.")
            print("PHOTO LOAD ERROR:", e)
            return

        if message.media_group_id not in media_groups:
            media_groups[message.media_group_id] = {
                "user_id": user_id,
                "message": message,
                "images": [],
                "caption": text,
            }

        media_groups[message.media_group_id]["images"].append(image_base64)

        if text:
            media_groups[message.media_group_id]["caption"] = text

        if message.media_group_id not in media_group_tasks:
            media_group_tasks[message.media_group_id] = asyncio.create_task(
                process_media_group(message.media_group_id)
            )

        return

    if message.voice:
        await safe_answer(message, "Слушаю голосовое...")
        try:
            text = await transcribe_voice(message)
        except Exception as e:
            await safe_answer(message, "Не получилось разобрать голосовое. Попробуйте ещё раз или напишите текстом.")
            print("VOICE TRANSCRIBE ERROR:", e)
            return

        if not text:
            await safe_answer(message, "Не получилось разобрать голосовое. Попробуйте ещё раз или напишите текстом.")
            return

        await safe_answer(message, "Понял вас. Минуточку...")

    if text and has_red_flag(text):
        await safe_answer(
            message,
            "Это похоже на тревожный симптом.\n\n"
            "Лучше не ждать ответа в чате, а срочно связаться с лечащим врачом "
            "или вызвать скорую помощь, особенно если состояние ухудшается."
        )
        return

    if not message.voice:
        await safe_answer(message, "Минуточку...")

    try:
        image_base64_list = None

        if message.photo:
            image_base64 = await photo_to_base64(message)
            image_base64_list = [image_base64]

            if not text:
                text = (
                    "Разберите это фото медицинского документа. "
                    "Выделите важные данные, объясните простыми словами и скажите, что нужно уточнить у врача."
                )

        elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            image_base64 = await image_document_to_base64(message)
            image_base64_list = [image_base64]

            if not text:
                text = (
                    "Разберите это изображение медицинского документа. "
                    "Выделите важные данные, объясните простыми словами и скажите, что нужно уточнить у врача."
                )

        elif not text:
            await safe_answer(message, "Пока я понимаю текст, голосовые и фото документов.")
            return

        print("SENDING TO AI")
        answer = await analyze_with_ai(user_id, text, image_base64_list)
        print("AI RESPONSE READY")

        await send_long_message(message, answer)

    except Exception as e:
        await safe_answer(
            message,
            "Сейчас не получилось получить ответ.\n\n"
            "Возможные причины: временный сбой, лимит API или проблема с подключением."
        )
        print("ERROR:", e)


async def main():
    print("Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
