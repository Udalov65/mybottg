import os
import re
import json
import asyncio
import psycopg2
import edge_tts
from io import BytesIO
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (Application, MessageHandler, CommandHandler,
                          CallbackQueryHandler, filters, ContextTypes)
import google.generativeai as genai
from groq import Groq
from tavily import TavilyClient


# === Ключи берутся из переменных окружения Railway ===
GEMINI_KEY = os.environ["GEMINI_KEY"]
GROQ_KEY = os.environ["GROQ_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OWNER_ID = int(os.environ["OWNER_ID"])
DATABASE_URL = os.environ["DATABASE_URL"]
TAVILY_KEY = os.environ.get("TAVILY_KEY", "")
ALLOWED = [int(x) for x in os.environ.get("ALLOWED", "").replace(" ", "").split(",") if x]
# =====================================================

SYSTEM_PROMPT = """Ты ИИ-ассистент с чувством юмора. Общайся на ты, дерзко и смешно.Шути, подкалывай, отвечай с сарказмом, можешь крепко выражаться и материться,черный юмор,
все свои, но делай это не огромными текстами, а коротко. И при этом реально помогай: за шутками не забывай нормально отвечать на вопрос.
Подстраивайся под собеседника и запоминай детали разговора."""

MAX_TURNS = 20
SUMMARY_TRIGGER = 30
VOICE_NAME = "ru-RU-DmitryNeural"  # мужской; женский: ru-RU-SvetlanaNeural

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite", system_instruction=SYSTEM_PROMPT)
groq_client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None


# --- База данных ---
def db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS memory (uid TEXT PRIMARY KEY, history TEXT, summary TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS users (uid BIGINT PRIMARY KEY, voice INTEGER DEFAULT 0)")
        cur.execute("ALTER TABLE memory ADD COLUMN IF NOT EXISTS summary TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS voice INTEGER DEFAULT 0")
        conn.commit()


def get_memory(uid):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT history, summary FROM memory WHERE uid = %s", (uid,))
        row = cur.fetchone()
        if not row:
            return [], ""
        return json.loads(row[0]), (row[1] or "")


def save_memory(uid, history, summary):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memory (uid, history, summary) VALUES (%s, %s, %s) "
            "ON CONFLICT (uid) DO UPDATE SET history = EXCLUDED.history, summary = EXCLUDED.summary",
            (uid, json.dumps(history, ensure_ascii=False), summary),
        )
        conn.commit()


def clear_memory(uid):
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM memory WHERE uid = %s", (uid,))
        conn.commit()


def add_user(uid):
    with db() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users (uid) VALUES (%s) ON CONFLICT DO NOTHING", (uid,))
        conn.commit()


def all_users():
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT uid FROM users")
        return [r[0] for r in cur.fetchall()]


def get_voice_mode(uid):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT voice FROM users WHERE uid = %s", (uid,))
        row = cur.fetchone()
        return bool(row[0]) if row else False


def set_voice_mode(uid, on):
    with db() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users (uid, voice) VALUES (%s, %s) "
                    "ON CONFLICT (uid) DO UPDATE SET voice = EXCLUDED.voice", (uid, 1 if on else 0))
        conn.commit()


# --- Выжимка памяти ---
def make_summary(old_summary, messages):
    text = "\n".join(f'{m["role"]}: {" ".join(m["parts"])}' for m in messages)
    prompt = ("Вот предыдущая выжимка о пользователе и кусок диалога. "
              "Обнови выжимку: сохрани важные факты (имя, интересы, договорённости). Коротко, без воды.\n\n"
              f"Старая выжимка:\n{old_summary}\n\nДиалог:\n{text}")
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception:
            return old_summary


# --- Groq резерв ---
def ask_groq(history, summary, text):
    sys = SYSTEM_PROMPT + (f"\n\nЧто ты помнишь о собеседнике: {summary}" if summary else "")
    messages = [{"role": "system", "content": sys}]
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": " ".join(m["parts"])})
    messages.append({"role": "user", "content": text})
    resp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    return resp.choices[0].message.content


# --- Чистим текст для озвучки (убираем эмодзи и разметку) ---
def clean_for_voice(text):
    text = re.sub(r"[^\w\s.,!?;:()«»\"'-]", " ", text)
    text = text.replace("*", " ").replace("#", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Озвучка через Edge TTS ---
async def make_voice(text, path):
    snippet = clean_for_voice(text)[:1000]
    communicate = edge_tts.Communicate(snippet, voice=VOICE_NAME)
    await communicate.save(path)


# --- Распознавание голоса ---
async def transcribe_voice(update, context):
    voice = update.message.voice
    tg_file = await context.bot.get_file(voice.file_id)
    ogg_path = f"/tmp/{voice.file_id}.ogg"
    await tg_file.download_to_drive(ogg_path)
    try:
        with open(ogg_path, "rb") as f:
            tr = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo", file=(ogg_path, f.read()))
        return tr.text
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)


# --- Распознавание кружков ---
async def transcribe_video_note(update, context):
    note = update.message.video_note
    tg_file = await context.bot.get_file(note.file_id)
    mp4_path, mp3_path = f"/tmp/{note.file_id}.mp4", f"/tmp/{note.file_id}.mp3"
    await tg_file.download_to_drive(mp4_path)
    try:
        os.system(f"ffmpeg -i {mp4_path} -vn -acodec libmp3lame {mp3_path} -y -loglevel quiet")
        with open(mp3_path, "rb") as f:
            tr = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo", file=(mp3_path, f.read()))
        return tr.text
    finally:
        for p in (mp4_path, mp3_path):
            if os.path.exists(p):
                os.remove(p)


# --- Распознавание изображений (со сжатием для экономии) ---
async def describe_image(update, context, caption):
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    raw = await tg_file.download_as_bytearray()
    img = Image.open(BytesIO(bytes(raw)))
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()
    question = caption or "Что на этом изображении? Опиши в своём стиле."
    r = model.generate_content([
        question,
        {"mime_type": "image/jpeg", "data": img_bytes},
    ])
    return r.text


# --- Получить ответ модели (Gemini → Groq) ---
async def get_answer(update, user, history, summary, user_text):
    sys = SYSTEM_PROMPT + (f"\n\nЧто ты помнишь о собеседнике: {summary}" if summary else "")
    local_model = genai.GenerativeModel("gemini-3.1-flash-lite", system_instruction=sys)
    chat = local_model.start_chat(history=history)
    warned = False
    for attempt in range(2):
        try:
            r = chat.send_message(user_text)
            new_history = [{"role": m.role, "parts": [p.text for p in m.parts]} for m in chat.history]
            print(f"[→ {user.first_name} | Gemini]: {r.text}")
            return r.text, new_history
        except Exception as e:
            if "429" in str(e):
                if not warned:
                    await update.message.reply_text("⏳ Сейчас много запросов, подожди пару секунд...")
                    warned = True
                if attempt < 1:
                    await asyncio.sleep(15)
                    continue
            print(f"Gemini не смог ({e}), пробую Groq...")
            break
    try:
        ans = ask_groq(history, summary, user_text)
        new_history = history + [{"role": "user", "parts": [user_text]},
                                 {"role": "model", "parts": [ans]}]
        print(f"[→ {user.first_name} | Groq]: {ans}")
        return ans, new_history
    except Exception as e:
        print(f"Groq тоже не смог: {e}")
        return None, history


# --- Отправка ответа: текст и голос подряд, без паузы ---
async def send_answer(update, context, uid, answer):
    if get_voice_mode(uid):
        path = f"/tmp/tts_{uid}.mp3"
        voice_ready = False
        try:
            await make_voice(answer, path)
            voice_ready = True
        except Exception as e:
            print(f"Озвучка не удалась: {e}")
        await update.message.reply_text(answer)
        if voice_ready:
            try:
                with open(path, "rb") as f:
                    await update.message.reply_voice(f)
                os.remove(path)
            except Exception as e:
                print(f"Отправка голоса не удалась: {e}")
    else:
        await update.message.reply_text(answer)


# --- Основной обработчик ---
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = str(user.id)
    add_user(user.id)

    if update.message.voice:
        try:
            user_text = await transcribe_voice(update, context)
            print(f"[{user.first_name} 🎤→]: {user_text}")
            await update.message.reply_text(f"🎤 Расшифровал: «{user_text}»")
        except Exception as e:
            await update.message.reply_text("Не смог разобрать голосовое 😕")
            print(f"Ошибка распознавания: {e}")
            return
    elif update.message.video_note:
        try:
            user_text = await transcribe_video_note(update, context)
            print(f"[{user.first_name} ⭕→]: {user_text}")
            await update.message.reply_text(f"⭕ Расшифровал: «{user_text}»")
        except Exception as e:
            await update.message.reply_text("Не смог разобрать кружок 😕")
            print(f"Ошибка распознавания кружка: {e}")
            return
    elif update.message.photo:
        if ALLOWED and user.id not in ALLOWED:
            await update.message.reply_text(f"Доступ только для семьи. Твой ID: {user.id}")
            return
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        print(f"[{user.first_name} 🖼→]: (фото)")
        try:
            answer = await describe_image(update, context, update.message.caption)
        except Exception as e:
            await update.message.reply_text("Не смог разобрать картинку 😕")
            print(f"Ошибка распознавания фото: {e}")
            return
        await send_answer(update, context, uid, answer)
        return
    else:
        user_text = update.message.text
        print(f"[{user.first_name} →]: {user_text}")

    if ALLOWED and user.id not in ALLOWED:
        await update.message.reply_text(
            f"Доступ только для семьи. Твой ID: {user.id}\nПередай его владельцу бота.")
        return

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    history, summary = get_memory(uid)
    history = history[-MAX_TURNS:]
    answer, new_history = await get_answer(update, user, history, summary, user_text)
    if answer is None:
        await update.message.reply_text("Оба сервиса сейчас заняты, попробуй через минуту 🙏")
        return

    await send_answer(update, context, uid, answer)

    if len(new_history) >= SUMMARY_TRIGGER:
        summary = make_summary(summary, new_history[:-MAX_TURNS])
        new_history = new_history[-MAX_TURNS:]
        print(f"[Память сжата для {user.first_name}]")
    save_memory(uid, new_history[-MAX_TURNS:], summary)


# --- Команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я ваш семейный ИИ-помощник 🤖\nПиши текстом, голосом, кружком или шли фото — отвечу на всё.\n"
        "Команды смотри в /help.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🧹 Очистить память", callback_data="reset")],
        [InlineKeyboardButton("🔊 Вкл/выкл озвучку", callback_data="voice")],
    ])
    await update.message.reply_text(
        "Что я умею:\n\n"
        "💬 Отвечаю на текст, 🎤 голосовые, ⭕ кружки и 🖼 фото\n"
        "🧠 Помню наш разговор\n"
        "🔊 /voice — отвечать ещё и голосом\n"
        "🔎 /search запрос — поиск свежего в интернете\n"
        "🧹 /reset — очистить память\n"
        "🆔 /myid — узнать свой ID",
        reply_markup=kb)


async def voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    new = not get_voice_mode(uid)
    set_voice_mode(uid, new)
    await update.message.reply_text("🔊 Озвучка включена!" if new else "🔇 Озвучка выключена.")


async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not tavily:
        await update.message.reply_text("Поиск не настроен (нет ключа Tavily).")
        return
    query = update.message.text.partition(" ")[2].strip()
    if not query:
        await update.message.reply_text("Напиши так: /search что ищем")
        return
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        res = tavily.search(query=query, max_results=5)
        found = "\n".join(f'- {r["title"]}: {r["content"][:200]}' for r in res["results"])
        prompt = (f"Вопрос: {query}\n\nСвежие данные из интернета:\n{found}\n\n"
                  "Ответь на вопрос на основе этих данных, кратко и по делу, в своём стиле.")
        ans = model.generate_content(prompt).text
    except Exception as e:
        await update.message.reply_text(f"Поиск не удался: {e}")
        return
    uid = str(update.effective_user.id)
    await send_answer(update, context, uid, ans)


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_memory(str(update.effective_user.id))
    await update.message.reply_text("Память очищена (включая выжимку).")


async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id != OWNER_ID:
        await update.message.reply_text("Рассылку может делать только владелец.")
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text("Напиши так: /broadcast текст")
        return
    sent, failed = 0, 0
    for target_id in all_users():
        if target_id == OWNER_ID:
            continue
        try:
            await context.bot.send_message(target_id, f"📢 Сообщение для семьи:\n\n{text}")
            sent += 1
        except Exception:
            failed += 1
    await update.message.reply_text(f"Готово. Отправлено: {sent}, не дошло: {failed}")


# --- Кнопки ---
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    uid = str(q.from_user.id)
    await q.answer()
    if q.data == "reset":
        clear_memory(uid)
        await q.edit_message_text("Память очищена 🧹")
    elif q.data == "voice":
        new = not get_voice_mode(uid)
        set_voice_mode(uid, new)
        await q.edit_message_text("🔊 Озвучка включена!" if new else "🔇 Озвучка выключена.")


# --- Запуск ---
init_db()
app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_cmd))
app.add_handler(CommandHandler("voice", voice_cmd))
app.add_handler(CommandHandler("search", search_cmd))
app.add_handler(CommandHandler("myid", myid))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CommandHandler("broadcast", broadcast))
app.add_handler(CallbackQueryHandler(on_button))
app.add_handler(MessageHandler(
    (filters.TEXT | filters.VOICE | filters.VIDEO_NOTE | filters.PHOTO) & ~filters.COMMAND, handle))
print("Бот запущен...")
app.run_polling()
