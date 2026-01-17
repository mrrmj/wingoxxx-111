from config import TELEGRAM_BOT_TOKEN

import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from prediction_model import predict_next_result
from user_manager import get_or_create_user, check_premium_status, set_premium_status
from data_manager import store_prediction, get_prediction_history
from functools import wraps
import time

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Dictionary to store last command usage time for each user
user_last_command_time = {}
RATE_LIMIT_SECONDS = 5  # Users can only send commands every 5 seconds

def rate_limit(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        now = time.time()
        if user_id in user_last_command_time and (now - user_last_command_time[user_id]) < RATE_LIMIT_SECONDS:
            remaining_time = int(RATE_LIMIT_SECONDS - (now - user_last_command_time[user_id]))
            await update.message.reply_text(f"Please wait {remaining_time} seconds before using this command again.")
            return
        user_last_command_time[user_id] = now
        return await func(update, context, *args, **kwargs)
    return wrapped


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    user = update.effective_user
    get_or_create_user(user.id)
    await update.message.reply_html(
        f"Hi {user.mention_html()}!\nWelcome to the toms WinGo30s Prediction Bot. Use /help to see available commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /help is issued."""
    await update.message.reply_text("Available Commands:\n/start - Start the bot\n/generate - Generate next prediction\n/history - Show recent predictions and actual results\n/stats - Display bot accuracy and performance metrics\n/help - Show usage instructions")


@rate_limit
async def generate_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates and shows the best recommended next prediction."""
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.message.reply_text("This feature is for premium users only. Please subscribe to access predictions.")
        return

    await update.message.reply_text("Generating prediction...")
    prediction = predict_next_result()
    if "error" in prediction:
        await update.message.reply_text(f"Error: {prediction["error"]}")
    else:
        message = (
            f"🎯 Next Period: #{prediction["period_number"]}\n"
            f"✅ Predicted Color: {prediction["color"]}\n"
            f"✅ Predicted Size: {prediction["size"]}\n"
            f"⚡️ Confidence: {prediction["confidence_score"]:.2f}%"
        )
        await update.message.reply_html(message)
        store_prediction(prediction["period_number"], prediction["color"], prediction["size"], prediction["confidence_score"])


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows recent predictions and actual results."""
    await update.message.reply_text("Fetching recent history...")
    history = get_prediction_history()
    if not history:
        await update.message.reply_text("No prediction history available.")
        return

    history_message = "Recent Predictions:\n"
    for record in history:
        history_message += (
            f"Period: {record.issueNumber}, Predicted: {record.predicted_color} {record.predicted_size}, "
            f"Confidence: {record.confidence_score:.2f}%, Actual: {record.actual_color or 'N/A'} {record.actual_size or 'N/A'}\n"
        )
    await update.message.reply_text(history_message)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays bot accuracy and performance metrics."""
    await update.message.reply_text("Calculating statistics...")
    # Placeholder for fetching and displaying stats
    await update.message.reply_text("Statistics feature is under development.")


def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # On different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("generate", generate_prediction))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("stats", stats_command))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


