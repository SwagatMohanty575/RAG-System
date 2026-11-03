import streamlit as st
import requests
import tempfile
import os

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #262626;
    }

    /* Conversation buttons: consistent spacing, rounded, full width */
    [data-testid="stSidebar"] .stButton button {
    width: 100%;
    height: 52px;
    border-radius: 20px;
    background-color: #3a3a3a;
    border: 1px solid #4a4a4a;
    color: #e8e8e8;
    text-align: left;
    padding: 10px 16px;
    margin-bottom: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: normal;
    line-height: 1.2;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #454545;
        border-color: #5a5a5a;
    }

    /* Delete (trash) buttons: small, no left-align */
    [data-testid="stSidebar"] .trash-btn button {
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    }


    .status-box {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 8px;
    }
    [data-testid="stHorizontalBlock"] {
    align-items: center;
    }
    [data-testid="stSidebar"] [data-testid="column"] {
    display: flex;
    align-items: center;
    }
    [data-testid="stSidebar"] .element-container {
    margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "restored" not in st.session_state:
    st.session_state.restored = True
    try:
        resp = requests.get(f"{API_BASE}/sessions/active", timeout=5).json()
        if resp.get("active"):
            st.session_state.active_session_id = resp["session_id"]
    except Exception:
        pass


def _history_to_messages(history):
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["human"]})
        messages.append({"role": "assistant", "content": turn["ai"], "sources": turn["sources"]})
    return messages


def activate_session(session_id: str):
    resp = requests.post(f"{API_BASE}/sessions/{session_id}/activate")
    data = resp.json()
    st.session_state.active_session_id = data["session_id"]
    st.session_state.messages = _history_to_messages(data["history"])


def delete_session(session_id: str):
    requests.delete(f"{API_BASE}/sessions/{session_id}")
    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = None
        st.session_state.messages = []


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Conversations")

    try:
        sessions = requests.get(f"{API_BASE}/sessions", timeout=5).json()["sessions"]
    except Exception:
        sessions = []

    if not sessions:
        st.caption("No chats yet.")

    for s in sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            is_active = s["session_id"] == st.session_state.active_session_id
            label = f"**{s['chat_title']}**" if is_active else s["chat_title"]
            if st.button(label, key=f"open_{s['session_id']}"):
                activate_session(s["session_id"])
                st.rerun()
        with col2:
            st.markdown('<div class="trash-btn">', unsafe_allow_html=True)
            if st.button("🗑", key=f"delete_{s['session_id']}"):
                delete_session(s["session_id"])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### Upload a document")
    st.caption("200MB per file • PDF, DOCX, XLSX, CSV, PPTX")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "csv", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.caption(f"📄 {uploaded_file.name}")

    process_clicked = st.button("Process", use_container_width=True, type="primary")

    if process_clicked:
        if not uploaded_file:
            st.warning("Choose a file first.")
        else:
            with st.spinner("Extracting, chunking, embedding..."):
                tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={"file_path": tmp_path},
                    timeout=2000,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['file_name']}")

                    active_resp = requests.get(f"{API_BASE}/sessions/active", timeout=5).json()
                    if active_resp.get("active"):
                        st.session_state.active_session_id = active_resp["session_id"]
                        st.session_state.messages = []
                    st.rerun()
                else:
                    st.error(f"Error: {resp.text}")

    st.divider()

    # Status indicators — plain Streamlit columns, no custom flex CSS
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        col1, col2 = st.columns(2)
        with col1:
            st.caption("NvidiaAPI")
            st.markdown("✅ OK" if health["api_ready"] else "❌ Down")
        with col2:
            st.caption("Vector DB")
            st.markdown("✅ OK" if health["vector_db_ready"] else "❌ Down")
        st.caption(f"Model: `{health['model']}`")
    except Exception:
        st.caption("Backend unreachable")


# ----------------------------------------------------------------------
# MAIN CHAT AREA
# ----------------------------------------------------------------------

st.title("Document Assistant")

if st.session_state.active_session_id is None:
    st.info("👈 Upload a document from the sidebar to start asking questions")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['file_name']}** — Page {src['page_number']}")
                        if src.get("section_title"):
                            st.caption(f"Section: {src['section_title']}")
                        st.code(src["chunk_preview"], language="text")

    prompt = st.chat_input("Ask anything")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.active_session_id,
                    },
                    timeout=1200,
                )

            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources", expanded=True):
                        for src in sources:
                            st.markdown(f"**{src['file_name']}** — Page {src['page_number']}")
                            if src.get("section_title"):
                                st.caption(f"Section: {src['section_title']}")
                            st.code(src["chunk_preview"], language="text")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                st.error(f"Error: {resp.text}")