from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# This list IS the memory. We grow it with every exchange.
messages = [
    {"role": "system", "content": "You are a helpful personal assistant. Be concise."}
]

print("🤖 Personal Assistant ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    print(user_input)
    
    if user_input.lower() == "quit":
        print("Bye!")
        break
    
    if not user_input:
        continue

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    # Send full history to GPT every time
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content

    # Add assistant reply to history
    messages.append({"role": "assistant", "content": assistant_reply})

    print(f"\n🤖 {assistant_reply}")
    print(f"   [tokens so far: {response.usage.total_tokens}]\n")