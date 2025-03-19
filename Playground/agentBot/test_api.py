import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"API key found: {'Yes' if api_key else 'No'}")
print(f"API key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

# Configure OpenAI client
openai.api_key = api_key
openai.api_base = "https://api.deepseek.com"  # Make sure this is the correct base URL

# Test connection
try:
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello, can you hear me?"}],
        temperature=0
    )
    print("API connection successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error connecting to API: {str(e)}")
    print("\nPossible solutions:")
    print("1. Check if your API key is correct")
    print("2. Verify the API base URL is correct (https://api.deepseek.com)")
    print("3. Make sure your DeepSeek account is active")
    print("4. Check if your IP is allowed to access the API")
