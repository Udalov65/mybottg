import os
import json
import asyncio
import psycopg2
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import google.generativeai as genai
from groq import Groq

# === Ключи берутся из переменных окружения Railway ===
GEMINI_KEY = os.environ["GEMINI_KEY"]
GROQ_KEY = os.environ["GROQ_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OWNER_ID = int(os.environ["OWNER_ID"])
DATABASE_URL = os.environ["DATABASE_URL"]
# =====================================================

# Список разрешённых: можно задать через переменную ALLOWED (ID через запятую),
# либо оставить пусто — тогда бот отвечает всем.
ALLOWED = [int(x) for x in os.environ.get("ALLOWED", "").replace(" ", "").split(",") if x]

SYSTEM_PROMPT = """Ты ИИ-ассистент с чувством юмора. Общайся на ты, дерзко и смешно.Шути, подкалывай, отвечай с сарказмом, можешь крепко выражаться и материться,черный юмор,
все свои, но делай это не огромными текстами, а коротко. И при этом реально помогай: за шутками не забывай нормально отвечать на вопрос.
Подстраивайся под собеседника и запоминай детали разговора."""

MAX_TURNS = 20          # сколько последних сообщений держать дословно
SUMMARY_TRIGGER = 30    # при скольких сообщениях сжимать старое в выжимку

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite", system_instruction=SYSTEM_PROMPT)
groq_client = Groq(api_key=GROQ_KEY)


# --- База данных ---
def db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS memory (uid TEXT PRIMARY KEY, history TEXT, summary TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS users (uid BIGINT PRIMARY KEY)")
        cur.execute("ALTER TABLE memory ADD COLUMN IF NOT EXISTS summary TEXT")
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


# --- Сжатие старой истории в выжимку ---
def make_summary(old_summary, messages):
    text = "\n".join(f'{m["role"]}: {" ".join(m["parts"])}' for m in messages)
    prompt = (
        "Вот предыдущая выжимка о пользователе и кусок диалога. "
        "Обнови выжимку: сохрани важные факты о человеке (имя, интересы, детали, договорённости). "
        "Коротко, по делу, без воды.\n\n"
        f"Старая выжимка:\n{old_summary}\n\nДиалог:\n{text}"
    )
    try:
        r = model.generate_content(prompt)
        return r.text.strip()
    except Exception:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return old_summary


# --- Запасной провайдер: Groq (для основного ответа) ---
def ask_groq(history, summary, text):
    sys = SYSTEM_PROMPT
    if summary:
        sys += f"\n\nЧто ты помнишь о собеседнике: {summary}"
    messages = [{"role": "system", "content": sys}]
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": " ".join(m["parts"])})
    messages.append({"role": "user", "content": text})
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return resp.choices[0].message.content


# --- Распознавание голосовых через Groq Whisper ---
async def transcribe_voice(update, context):
    voice = update.message.voice
    tg_file = await context.bot.get_file(voice.file_id)
    ogg_path = f"/tmp/{voice.file_id}.ogg"
    await tg_file.download_to_drive(ogg_path)
    try:
        with open(ogg_path, "rb") as f:
            tr = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(ogg_path, f.read()),
            )
        return tr.text
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)


# --- Распознавание кружков (видеосообщений) ---
async def transcribe_video_note(update, context):
    note = update.message.video_note
    tg_file = await context.bot.get_file(note.file_id)
    mp4_path = f"/tmp/{note.file_id}.mp4"
    mp3_path = f"/tmp/{note.file_id}.mp3"
    await tg_file.download_to_drive(mp4_path)
    try:
        os.system(f"ffmpeg -i {mp4_path} -vn -acodec libmp3lame {mp3_path} -y -loglevel quiet")
        with open(mp3_path, "rb") as f:
            tr = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(mp3_path, f.read()),
            )
        return tr.text
    finally:
        for p in (mp4_path, mp3_path):
            if os.path.exists(p):
                os.remove(p)


# --- Основной обработчик ---
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = str(user.id)
    add_user(user.id)

    # Текст, голос или кружок?
    if update.message.voice:
        try:
            user_text = await transcribe_voice(update, context)
            print(f"[{user.first_name} 🎤→]: {user_text}")
            await update.message.reply_text(f"🎤 Расшифровал: «{user_text}»")
        except Exception as e:
            await update.message.reply_text("Не смог разобрать голосовое 😕 попробуй ещё раз или текстом.")
            print(f"Ошибка распознавания: {e}")
            return
    elif update.message.video_note:
        try:
            user_text = await transcribe_video_note(update, context)
            print(f"[{user.first_name} ⭕→]: {user_text}")
            await update.message.reply_text(f"⭕ Расшифровал: «{user_text}»")
        except Exception as e:
            await update.message.reply_text("Не смог разобрать кружок 😕 попробуй ещё раз или текстом.")
            print(f"Ошибка распознавания кружка: {e}")
            return
    else:
        user_text = update.message.text
        print(f"[{user.first_name} →]: {user_text}")

    if ALLOWED and user.id not in ALLOWED:
        await update.message.reply_text(
            f"Доступ только для семьи. Твой ID: {user.id}\n"
            "Передай его владельцу бота, чтобы он тебя добавил."
        )
        return

    history, summary = get_memory(uid)
    history = history[-MAX_TURNS:]

    sys = SYSTEM_PROMPT
    if summary:
        sys += f"\n\nЧто ты помнишь о собеседнике: {summary}"
    local_model = genai.GenerativeModel("gemini-3.1-flash-lite", system_instruction=sys)

    answer = None
    warned = False
    chat = local_model.start_chat(history=history)
    for attempt in range(2):
        try:
            r = chat.send_message(user_text)
            answer = r.text
            new_history = [
                {"role": m.role, "parts": [p.text for p in m.parts]}
                for m in chat.history
            ]
            print(f"[→ {user.first_name} | Gemini]: {answer}")
            break
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

    if answer is None:
        try:
            answer = ask_groq(history, summary, user_text)
            new_history = history + [
                {"role": "user", "parts": [user_text]},
                {"role": "model", "parts": [answer]},
            ]
            print(f"[→ {user.first_name} | Groq]: {answer}")
        except Exception as e:
            await update.message.reply_text("Оба сервиса сейчас заняты, попробуй через минуту 🙏")
            print(f"Groq тоже не смог: {e}")
            return

    await update.message.reply_text(answer)

    # Если история разрослась — сжимаем старую часть в выжимку
    if len(new_history) >= SUMMARY_TRIGGER:
        old_part = new_history[:-MAX_TURNS]
        keep = new_history[-MAX_TURNS:]
        summary = make_summary(summary, old_part)
        new_history = keep
        print(f"[Память сжата для {user.first_name}]")

    save_memory(uid, new_history[-MAX_TURNS:], summary)


# --- Команды ---
async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_memory(str(update.effective_user.id))
    await update.message.reply_text("Память диалога очищена (включая выжимку).")


async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id != OWNER_ID:
        await update.message.reply_text("Рассылку может делать только владелец.")
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        await update.message.reply_text("Напиши так: /broadcast текст сообщения")
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


# --- Запуск ---
init_db()
app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("myid", myid))
app.add_handler(CommandHandler("reset", reset))
app.add_handler(CommandHandler("broadcast", broadcast))
app.add_handler(MessageHandler((filters.TEXT | filters.VOICE | filters.VIDEO_NOTE) & ~filters.COMMAND, handle))
print("Бот запущен...")
app.run_polling()
