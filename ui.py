import streamlit as st
import textwrap
from llm import create_db_from_youtube_video_url, get_response_from_query

st.title("Youtube ChatMate")

# Sidebar to display history
history_sidebar = st.sidebar.empty()


def startChat():
    video_url = st.text_input("Enter YouTube Video URL", value="")

    if video_url:
        query = st.text_input("What do you want to know about this video?", value="")

        if len(query) == 0:
            query = "Describe this video"

        if st.button("Describe Video"):
            db = create_db_from_youtube_video_url(video_url=video_url)
            response, docs = get_response_from_query(db, query)
            st.write(textwrap.fill(response, width=50))

            # Update history in the sidebar
            history_sidebar.text(f"Query: {query}")
            history_sidebar.text("Response:")
            history_sidebar.text(response)

            if st.button("New Chat"):
                # Clear text inputs
                st.text_input("Enter YouTube Video URL", value="")
                st.text_input("What do you want to know about this video?", value="")

                startChat()


if __name__ == "__main__":
    startChat()
