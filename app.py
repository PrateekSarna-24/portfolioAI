from __future__ import annotations
import streamlit as st
import base64
import json
from datetime import date
from pathlib import Path
from typing import Annotated
from urllib.parse import quote_plus
import os

from PIL import Image
from pydantic import BaseModel, EmailStr, Field, ValidationError
from streamlit_option_menu import option_menu

from dotenv import load_dotenv

# LangChain / ChatTogether imports
from langchain_core.messages import SystemMessage, HumanMessage  # ensure this import path is correct for your version
from langchain_together import ChatTogether

# ========================= Page config & global CSS ========================= #
st.set_page_config(
    page_title="Prateek Sarna — Portfolio",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="collapsedControl"] { display: none; }
      footer { visibility: hidden; }
      .profile-pic { width: 220px; height: 220px; border-radius: 50%; object-fit: cover; border: 3px solid rgba(0,0,0,0.08); }
      .center { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================ Assets =================================== #
AVATAR_PATHS = ["prateek.jpg", "avatar.png"]
avatar_b64 = None
for p in AVATAR_PATHS:
    if Path(p).exists():
        with open(p, "rb") as f:
            avatar_b64 = base64.b64encode(f.read()).decode()
        break

RESUME_PATH = Path("PRATEEK_SARNA_RESUME_2025.pdf")
resume_pdf_bytes = RESUME_PATH.read_bytes() if RESUME_PATH.exists() else None

# ============================== Sidebar Nav ================================ #
with st.sidebar:
    st.markdown(
        """
        <div class='center'>
            <h2 style='margin-bottom:0'>Prateek Sarna</h2>
            <p style='margin-top:4px; opacity:0.8'>Developer • ML & Data • System Design</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if avatar_b64:
        st.markdown(
            f"""
            <div class='center'>
                <img src='data:image/png;base64,{avatar_b64}' class='profile-pic' alt='Prateek Sarna'>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='center' style='font-size:48px'>🧠</div>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Navigate",
        options=[
            "Home",
            "AI Portfolio Chatbot",
            "Skills",
            "Projects",
            "Experience",
            "Contact",
            "Resume",
        ],
        icons=["house", "robot", "graph-up", "kanban", "briefcase", "envelope", "file-earmark-person"],
        default_index=0,
        orientation="vertical",
    )

# ================================= Home ==================================== #
if selected == "Home":
    left, right = st.columns([1, 2], vertical_alignment="center")

    with left:
        if avatar_b64:
            st.markdown(
                f"""
                <div class='center'>
                    <img src='data:image/png;base64,{avatar_b64}' class='profile-pic' alt='Prateek Sarna'>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.metric("CGPA", "8.16")
        st.metric("Projects", "15+")
        st.metric("DSA", "700+")

    with right:
        st.title("Building things that learn — and last.")
        st.write(
            """
            I'm **Prateek Sarna**, a software developer focused on **DSA**, **ML/DL**, and **system design**. I build
            **end-to-end platforms**: frontend (Streamlit/HTML), backend (**FastAPI**), reliable **REST APIs**, and
            containerized deployments with **Docker**. I also orchestrate AI pipelines with **LangChain** where useful.

            **Engineering Stack**
            - **Languages**: C++, Python, Java, SQL, HTML/CSS/JS
            - **AI/ML**: scikit-learn, TensorFlow/Keras, CNN/RNN/Transformers, multimodal fusion, LangChain
            - **Backend**: FastAPI, auth, rate-limiting, logging, telemetry
            - **System Design**: microservices, caching, pagination, idempotency, retries, observability
            - **DevOps**: Docker, Git, CI/CD, Unix; reproducible builds & envs
            - **Data**: Pandas/NumPy, SQL, data validation, ETL
            """
        )
        st.info(
            "Currently exploring: Full end-to-end development with **LangChain**, **FastAPI**, and **Docker** — building robust APIs, orchestration, and deployable AI services."
        )

# ================================ Skills =================================== #
elif selected == "Skills":
    st.title("Skills & Depth")
    st.caption("From idea → architecture → implementation → deploy → observe.")

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Programming")
        st.write("C++, Python, Java, SQL, HTML/CSS/JS")
        st.subheader("AI/ML/DL")
        st.write("scikit-learn, TensorFlow/Keras, CNN, RNN, Transformers, LangChain, vector DBs")
    with cols[1]:
        st.subheader("Backend & APIs")
        st.write("FastAPI, REST design, auth, rate-limits, pagination, versioning, OpenAPI")
        st.subheader("Data")
        st.write("Pandas, NumPy, SQL, ETL, validation")
    with cols[2]:
        st.subheader("Systems & DevOps")
        st.write("System design, Docker, Git, CI/CD, Unix, logging, monitoring")

# ================================ Projects ================================= #
elif selected == "Projects":

    from streamlit_extras.card import card  # optional for better visuals

    def tech_badge(name, color="#4CAF50"):
        return f"<span style='background-color:{color}; color:white; padding:4px 10px; border-radius:12px; font-size:0.85em; margin:2px; display:inline-block;'>{name}</span>"

    st.title("🚀 Projects")
    st.markdown("Here are some of the projects I’ve worked on, showcasing my skills in **AI, Deep Learning, Automation, and Full-Stack AI apps**:")

    projects = [
        {
            "title": "Medical Summarizer (LangChain + FastAPI + Docker)",
            "desc": "Built an <b>end-to-end AI application</b> for summarizing patient records using <b>LangChain</b>, deployed with <b>FastAPI</b> backend and containerized in <b>Docker</b>. This project highlights my ability to build production-ready AI pipelines.",
            "tech": [
                ("LangChain", "#FF9800"),
                ("FastAPI", "#009688"),
                ("Docker", "#0db7ed"),
                ("Python", "#3572A5")
            ],
            "link": None
        },
        {
            "title": "Neural Machine Translation",
            "desc": "Developed a <b>French-to-English NMT system</b> with an <b>encoder–decoder architecture</b> and <b>Bahdanau attention mechanism</b>. The model dynamically focuses on key input words during translation, improving fluency and handling longer sentences. Trained on French-English parallel datasets and evaluated with <b>BLEU scores</b>.",
            "tech": [
                ("Python", "#3572A5"),
                ("TensorFlow", "#FF6F00"),
                ("Keras", "#D00000")
            ],
            "link": "https://github.com/PrateekSarna-24/Neural-Machine-Translation"
        },
        {
            "title": "InstaBot Automation",
            "desc": "An <b>Instagram automation tool</b> built with <b>Selenium</b>, capable of scraping data and automating user interactions (likes, comments, follows). Helped me explore <b>API limitations and automation techniques</b>.",
            "tech": [
                ("Python", "#3572A5"),
                ("Selenium", "#43A047"),
                ("API Integration", "#795548")
            ],
            "link": "https://github.com/PrateekSarna-24/InstaBot"
        },
        {
            "title": "Pneumonia Detection",
            "desc": "Designed a <b>deep learning CNN model</b> for detecting pneumonia from chest X-ray images. Preprocessed medical images, trained on labeled datasets, and achieved strong accuracy in classification. This project combines <b>computer vision</b> with <b>healthcare AI</b>.",
            "tech": [
                ("Python", "#3572A5"),
                ("TensorFlow", "#FF6F00"),
                ("CNN", "#9C27B0")
            ],
            "link": None
        }
    ]

    # Sort LangChain projects first
    projects.sort(key=lambda p: "LangChain" not in [t[0] for t in p["tech"]])

    # Display in 2-column grid
    cols = st.columns(2)
    for idx, proj in enumerate(projects):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div style='background-color:#f2f2f7; padding:18px; border-radius:15px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:20px;'>
                    <h3 style='margin-bottom:5px; color:#222;'>{proj["title"]}</h3>
                    <p style='font-size:0.9em; color:#444; line-height:1.4;'>{proj["desc"]}</p>
                    <div>{"".join([tech_badge(name, color) for name, color in proj["tech"]])}</div>
                    <br>
                    {f"<a href='{proj['link']}' target='_blank' style='text-decoration:none; background:#4CAF50; color:white; padding:6px 12px; border-radius:8px;'>🔗 View Project</a>" if proj["link"] else "<span style='color:gray; font-size:0.9em;'>🚧 Coming Soon</span>"}
                </div>
                """,
                unsafe_allow_html=True
            )

# =============================== Experience ================================ #
elif selected == "Experience":
    st.title("Experience & Leadership")
    st.markdown(
        """
        - **Team Lead** — Led ML-focused teams; reviews, sprints, delivery.
        - **Hackathon** — Live video+audio analysis; pivoted to PyTorch under time pressure.
        - **MLOps** — From notebooks to reproducible deployments with Streamlit & Docker.
        """
    )
    st.markdown("---")
    st.subheader("Academic")
    st.write("CSE Graduate • CGPA **8.16** • NPTEL: Top 1% (Discrete Math), Top 2% (Probability & Statistics)")

# ================================ Contact ================================== #
elif selected == "Contact":
    class ContactForm(BaseModel):
        name: str = Field(..., min_length=2, max_length=50)
        email: EmailStr
        message: str = Field(..., min_length=2, max_length=500)

    st.set_page_config(page_title="Contact - Prateek Sarna", layout="centered")
    st.title("📬 Contact Me")
    st.write("Fill in your details below, and when you click **Send**, WhatsApp will open with your message ready to send to me.")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send via WhatsApp")

    if submitted:
        try:
            form = ContactForm(name=name, email=email, message=message)
            phone_number = "+919911400546"
            encoded_msg = quote_plus(f"Hello, my name is {form.name} ({form.email}).\n\n{form.message}")
            whatsapp_url = f"https://wa.me/{phone_number}?text={encoded_msg}"
            st.success("Validation successful! Opening WhatsApp...")
            st.markdown(f"[Click here if WhatsApp doesn't open automatically]({whatsapp_url})", unsafe_allow_html=True)
            st.markdown(f"<meta http-equiv='refresh' content='0; url={whatsapp_url}'>", unsafe_allow_html=True)
        except ValidationError as e:
            st.error("Please correct the errors below:")
            for err in e.errors():
                st.write(f"**{err['loc'][0]}**: {err['msg']}")

# ================================= Resume ================================== #
elif selected == "Resume":
    st.title("Resume & Downloads")
    st.write("Grab my latest resume/CV or request a tailored version for a specific role.")
    if resume_pdf_bytes:
        st.download_button(
            label="Download Resume (PDF)",
            data=resume_pdf_bytes,
            file_name="Prateek_Sarna_Resume.pdf",
            mime="application/pdf",
        )
    else:
        st.write("🛈 Resume file not found.")

    st.caption("Resume")
    st.markdown("---")
    st.subheader("Quick Snapshot")
    st.write(
        """
        - **Roles of interest**: Application Engineering, Software Development (SDE), ML Engineering.  
        - **Keywords**: Python, DSA, ML/DL, Unix, Docker, Git, Streamlit, SQL, APIs, Data Pipelines.  
        - **Open to**: Internships, part-time roles, and research collaborations.
        """
    )

# ================================== AI Portfolio Chatbot ====================
elif selected == "AI Portfolio Chatbot":
    st.title("🤖 AI Portfolio Chatbot")
    st.markdown("Ask me anything about **Prateek Sarna's** skills, projects, and experience!")

    # System prompt
    system_prompt = r"""
You are an AI portfolio assistant for Prateek Sarna, B.E. CSE student (Chandigarh University, 2021–2025, CGPA 8.16).

Summary:
- Skilled in software dev, data science, AI, and system design; 700+ DSA problems solved.
- Research: Deep learning emotion recognition (98%+ accuracy, IEEE 2023).

Projects (auto-explain when mentioned):
...
Rules:
1. Only answer queries about Prateek Sarna’s professional portfolio.
2. For unrelated queries, respond: "I’m here to assist with queries about Prateek’s professional portfolio only."
3. Format responses in **bullet points** or **tables** for clarity and professionalism.
4. But be warm in your conversation, do not sound like robot telling facts, be a little pleasant.

Also explain everything within 200 words.
"""

    api_key = st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY"))

    model = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        max_tokens=350,
        together_api_key=api_key
    )

    def invoke_query(user_query: str):
        chat_history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        try:
            result = model.invoke(chat_history)
            return getattr(result, "content", str(result))
        except Exception as e:
            return f"❌ API call failed: {e}"

    user_input = st.text_input("Type your message…")

    if st.button("Send") and user_input.strip():
        ai_response = invoke_query(user_input)
        st.markdown(f"**AI:** {ai_response}")

# ================================= Footer ================================== #
st.markdown(
    f"""
    <hr style='margin-top:3rem;margin-bottom:0.5rem' />
    <div style='display:flex;justify-content:space-between;align-items:center;opacity:0.8'>
      <div>© {date.today().year} Prateek Sarna</div>
      <div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
