import streamlit as st

from src.utils.api_utils import get_gemini_chatbot_response

st.set_page_config(page_title="AI Chatbot", layout="wide")

# Check if recommendation is available
if not st.session_state.get("recommendation"):
    st.warning("No recommendation found. Please run analysis from the home page.")
    st.stop()

# Layout: 2 Columns
col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("📋 Recommendation Summary")
    st.success(st.session_state["recommendation"])

with col2:
    st.subheader("🤖 Ask the AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    user_input = st.chat_input("Ask about your results...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    bot_msg = get_gemini_chatbot_response(
                        user_input,
                        st.session_state["recommendation"],
                        st.session_state.get("insect_count", 0),
                        st.session_state.get("flower_count", 0),
                        st.session_state.get("weather_data", {}),
                        st.session_state["chat_history"],
                    )
                except Exception as e:
                    bot_msg = "Sorry, something went wrong. Please try again later."
                    st.error(str(e))
                st.write(bot_msg)

        st.session_state.chat_history.append((user_input, bot_msg))

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
