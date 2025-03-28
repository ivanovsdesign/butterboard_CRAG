import os
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import dotenv_values

config = dotenv_values('../.env')

# Configure your bot
BOT_TOKEN = config['TELEGRAM_TOKEN']
JSONL_FILE = "../data/russian_crag.jsonl"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

def validate_json_structure(data: dict) -> None:
    """Validate JSON structure without content checks"""
    required_fields = {
        "interaction_id": str,
        "query_time": str,
        "domain": str,
        "question_type": str,
        "static_or_dynamic": str,
        "query": str,
        "answer": str,
        "alt_ans": list,
        "split": int,
        "search_results": list
    }
    
    # Check top-level fields
    for field, field_type in required_fields.items():
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(data[field], field_type):
            raise ValueError(f"Field {field} must be {field_type.__name__}")

    # Validate alt_ans
    if not all(isinstance(item, str) for item in data["alt_ans"]):
        raise ValueError("All alt_ans items must be strings")

    # Validate search_results
    required_result_fields = {
        "page_name": str,
        "page_url": str,
        "page_snippet": str,
        "page_result": str,
        "page_last_modified": str
    }
    
    for result in data["search_results"]:
        for field, field_type in required_result_fields.items():
            if field not in result:
                raise ValueError(f"Missing search result field: {field}")
            if not isinstance(result[field], field_type):
                raise ValueError(f"Search result field {field} must be {field_type.__name__}")

@dp.message(Command("start"))
async def start_command(message: Message):
    """Handle /start command"""
    await message.answer(
        "Send me JSON messages (either directly or forwarded) with the required structure. "
        "I'll validate and store them in the dataset."
    )

@dp.message(Command("help"))
async def help_command(message: Message):
    """Handle /help command"""
    await message.answer(
        "How to use:\n"
        "1. Send or forward messages containing JSON data\n"
        "2. The bot will validate the structure\n"
        "3. Valid messages will be saved to the dataset\n"
        "4. You'll receive immediate feedback about the operation"
    )

@dp.message()
async def handle_json_message(message: Message):
    """Handle all text messages containing JSON data"""
    try:
        # Attempt to parse JSON
        json_data = json.loads(message.text)
        
        # Validate structure
        validate_json_structure(json_data)
        
        # Save to JSONL file
        with open(JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        
        # Prepare response message
        response = "✅ JSON successfully saved to dataset!"
        if message.forward_from:
            response += f"\nForwarded from: @{message.forward_from.username}"
        elif message.forward_from_chat:
            response += f"\nForwarded from: {message.forward_from_chat.title}"
        
        await message.answer(response)
        
    except json.JSONDecodeError:
        await message.answer("❌ Invalid JSON format. Please send valid JSON data.")
    except ValueError as e:
        await message.answer(f"❌ Structural error: {str(e)}")
    except Exception as e:
        await message.answer(f"❌ Unexpected error: {str(e)}")

async def main():
    # Create JSONL file if not exists
    if not os.path.exists(JSONL_FILE):
        open(JSONL_FILE, "w").close()
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())