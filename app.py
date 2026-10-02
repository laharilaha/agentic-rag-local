import os
import tempfile
import gc
import base64

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

from src.agentic_rag.tools.custom_tool import DocumentSearchTool


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def load_llm():
    return LLM(
        model="ollama/qwen2.5-coder:7b",
        base_url="http://localhost:11434",
    )


# =========================================================
# CREATE CREW
# =========================================================

def create_crew(pdf_tool):
    llm = load_llm()

    retriever_agent = Agent(
        role="PDF Information Retriever",
        goal=(
            "Find the most relevant information from the uploaded PDF "
            "to answer the user's query."
        ),
        backstory=(
            "You are a careful document researcher. "
            "You must use the PDF search tool to find information "
            "before producing an answer."
        ),
        tools=[pdf_tool],
        llm=llm,
        verbose=True,
    )

    response_synthesizer_agent = Agent(
        role="Response Synthesizer",
        goal=(
            "Convert the retrieved PDF information into a clear, "
            "concise and accurate answer to the user's question."
        ),
        backstory=(
            "You are a precise technical communicator. "
            "Use the information retrieved from the document and "
            "do not invent facts."
        ),
        llm=llm,
        verbose=True,
    )

    retrieval_task = Task(
        description=(
            "Search the uploaded PDF for information relevant to "
            "this user query: {query}"
        ),
        expected_output=(
            "Relevant text retrieved from the uploaded PDF."
        ),
        agent=retriever_agent,
    )

    response_task = Task(
        description=(
            "Using the information retrieved by the previous task, "
            "answer this user query: {query}. "
            "If the document does not contain enough information, "
            "clearly say that the information was not found."
        ),
        expected_output=(
            "A concise and accurate answer based only on the "
            "retrieved document information."
        ),
        agent=response_synthesizer_agent,
    )

    crew = Crew(
        agents=[
            retriever_agent,
            response_synthesizer_agent,
        ],
        tasks=[
            retrieval_task,
            response_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("📚 Agentic RAG powered by CrewAI")
st.caption(
    "Chat with your PDF using semantic retrieval and a local Ollama LLM."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if uploaded_file is not None:

        if st.session_state.pdf_tool is None:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:

                with st.spinner("Indexing PDF..."):

                    st.session_state.pdf_tool = DocumentSearchTool(
                        file_path=temp_file_path
                    )

                st.session_state.crew = None

                st.success("PDF indexed successfully!")

            except Exception as e:

                st.error("PDF indexing failed.")

                st.exception(e)

            finally:

                # The document has already been indexed into
                # the in-memory Qdrant database.
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    if st.button("Clear Chat"):

        st.session_state.messages = []
        gc.collect()

        st.rerun()


# =========================================================
# SHOW PDF STATUS
# =========================================================

if st.session_state.pdf_tool is None:

    st.info(
        "Upload a PDF from the sidebar to start chatting."
    )

else:

    st.success(
        f"Ready to answer questions about: {uploaded_file.name}"
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Ask a question about your PDF..."
)


if prompt:

    if st.session_state.pdf_tool is None:

        st.warning(
            "Please upload a PDF before asking a question."
        )

        st.stop()


    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    # -----------------------------------------------------
    # CREATE CREW
    # -----------------------------------------------------

    if st.session_state.crew is None:

        with st.spinner("Preparing AI agents..."):

            st.session_state.crew = create_crew(
                st.session_state.pdf_tool
            )


    # -----------------------------------------------------
    # RUN CREW
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching the PDF and generating answer..."):

            try:

                result = st.session_state.crew.kickoff(
                    inputs={
                        "query": prompt
                    }
                )

                answer = result.raw

            except Exception as e:

                answer = (
                    "An error occurred while processing your question."
                )

                st.error(str(e))


        st.markdown(answer)


    # -----------------------------------------------------
    # SAVE RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )