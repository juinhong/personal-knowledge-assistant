import streamlit as st

st.title("🤖 Streamlit Basics Test")

# st.write — renders anything: text, markdown, data
st.write("Hello, this is **bold** and this is *italic*")
st.write({"key": "value", "number": 42})  # renders dicts as tables

# st.chat_message — renders a chat bubble
with st.chat_message("user"):
    st.write("Hey, what are Roaring Bitmaps?")

with st.chat_message("assistant"):
    st.write("Roaring Bitmaps are a compressed data structure for large sets of integers!")

# st.chat_input — the message input box at the bottom
user_input = st.chat_input("Type a message...")

if user_input:
    st.write(f"You typed: {user_input}")

# Session state persists across reruns
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Click me"):
    st.session_state.count += 1

st.write(f"Button clicked {st.session_state.count} times")

# Simulate conversation history
messages = [
    {"role": "user", "content": "What are Roaring Bitmaps?"},
    {"role": "assistant", "content": "A compressed data structure for integers."},
    {"role": "user", "content": "Where are they used?"},
    {"role": "assistant", "content": "Databases, search engines, and analytics."},
]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
