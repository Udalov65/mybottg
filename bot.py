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

MAX_TURNS = 20

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite", system_instruction=SYSTEM_PROMPT)
groq_client = Groq(api_key=GROQ_KEY)


# --- База данных ---
def db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS memory (uid TEXT PRIMARY KEY, history TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS users (uid BIGINT PRIMARY KEY)")
        conn.commit()


def get_history(uid):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT history FROM memory WHERE uid = %s", (uid,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else []


def save_history(uid, history):
    data = json.dumps(history, ensure_ascii=False)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO memory (uid, history) VALUES (%s, %s) "
            "ON CONFLICT (uid) DO UPDATE SET history = EXCLUDED.history",
            (uid, data),
        )
        conn.commit()


def clear_history(uid):
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


# --- Запасной провайдер: Groq ---
def ask_groq(history, text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": " ".join(m["parts"])})
    messages.append({"role": "user", "content": text})
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return resp.choices[0].message.content


# --- Основной обработчик ---
async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = str(user.id)
    add_user(user.id)
    print(f"[{user.first_name} →]: {update.message.text}")

    if ALLOWED and user.id not in ALLOWED:
        await update.message.reply_text(
            f"Доступ только для семьи. Твой ID: {user.id}\n"
            "Передай его владельцу бота, чтобы он тебя добавил."
        )
        return

    history = get_history(uid)[-MAX_TURNS:]
    answer = None
    warned = False

    chat = model.start_chat(history=history)
    for attempt in range(2):
        try:
            r = chat.send_message(update.message.text)
            answer = r.text
            new_history = [
                {"role": m.role, "parts": [p.text for p in m.parts]}
                for m in chat.history
            ][-MAX_TURNS:]
            print(f"[→ {user.first_name} | Gemini]: {answer}")
            break
        except Exception as e:
            if "429" in str(e):
                if not warned:
                    await update.message.reply_text(
                        "⏳ Сейчас много запросов, подожди пару секунд — обрабатываю..."
                    )
                    warned = True
                if attempt < 1:
                    await asyncio.sleep(15)
                    continue
            print(f"Gemini не смог ({e}), пробую Groq...")
            break

    if answer is None:
        try:
            answer = ask_groq(history, update.message.text)
            new_history = history + [
                {"role": "user", "parts": [update.message.text]},
                {"role": "model", "parts": [answer]},
            ]
            new_history = new_history[-MAX_TURNS:]
            print(f"[→ {user.first_name} | Groq]: {answer}")
        except Exception as e:
            await update.message.reply_text("Оба сервиса сейчас заняты, попробуй через минуту 🙏")
            print(f"Groq тоже не смог: {e}")
            return

    await update.message.reply_text(answer)
    save_history(uid, new_history)


# --- Команды ---
async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_history(str(update.effective_user.id))
    await update.message.reply_text("Память диалога очищена.")


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
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
print("Бот запущен...")
app.run_polling()