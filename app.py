import os
import base64
import random
from pathlib import Path
import chromadb
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, StorageContext,
                              PromptTemplate, QueryBundle)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# ---------- הנחיות ל-Gemini ----------
prompt = PromptTemplate(
    "מומחה לחוקי טיקט טו רייד. עברית, קצר, רק לפי הקטעים, בלי להזכיר אותם.\n"
    "- אין תשובה: 'אין לי מידע על זה'.\n"
    "- בלי מפה/חוק בהרחבה: חוקי הבסיס.\n"
    "- ציין מפה רק אם לא צוינה או נשאלה.\n"
    "- 'מה זה X': משפט-שניים, בלי חוקים.\n"
    "- 'מי אתה': אני עוזר לחוקי המשחק טיקט טו רייד.\n\n"
    "קטעים:\n{context_str}\n\nשאלה: {query_str}\nתשובה:"
)
condense = PromptTemplate(
    "שיחה:\n{history}\n\nהודעה: {question}\n\n"
    "אם זו לא שאלה (תודה, היי וכו'), החזר: שיחה | ותגובה קצרה ומנומסת.\n"
    "אחרת, החזר רק את השאלה מנוסחת כשאלה עצמאית ומלאה."
)
hyde = PromptTemplate(
    "כתוב פסקה קצרה מחוברת החוקים של טיקט טו רייד שעונה על: {question}"
)


# ---------- תווית מפה לפי שם הקובץ ----------
def get_edition(file_path):
    name = Path(file_path).stem.split("_", 1)[1]      # 04_europe -> europe
    return {"edition": "general" if name.startswith("general") else name}


# ---------- טעינת הבוט (פעם אחת בלבד) ----------
@st.cache_resource
def load_bot():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=api_key)
    llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=api_key)

    collection = chromadb.PersistentClient(path="./ttr_db").get_or_create_collection("ticket_to_ride")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() == 0:
        # המאגר ריק: קוראים את הקבצים, מחלקים ל-150 מילים (חפיפה 20) ושומרים
        documents = SimpleDirectoryReader("./ttr_rules", file_metadata=get_edition).load_data()
        splitter = SentenceSplitter(chunk_size=150, chunk_overlap=20, tokenizer=str.split)
        nodes = splitter.get_nodes_from_documents(documents)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    else:
        # המאגר כבר קיים: טוענים ממנו
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5, text_qa_template=prompt)
    return llm, query_engine


# ---------- מענה על שאלה ----------
def answer_question(question, history):
    # עיבוי: רק אם יש שיחה קודמת
    if history:
        question = llm.complete(condense.format(history="\n".join(history),
                                                question=question)).text.strip()
        # הודעה שאינה שאלה (תודה, הבנתי...) - עונים בקצרה בלי לחפש
        if question.startswith("שיחה"):
            return question.split("|", 1)[-1].strip()

    # HyDE: רק לשאלות קצרות (עד 4 מילים)
    if len(question.split()) <= 4:
        guess = llm.complete(hyde.format(question=question)).text.strip()
        query = QueryBundle(query_str=question, custom_embedding_strs=[guess, question])
    else:
        query = question

    answer = str(query_engine.query(query)).strip()
    history += [f"משתמש: {question}", f"בוט: {answer}"]
    return answer


# ---------- רקע בסגנון מפת רכבות (נבנה בקוד, SVG) ----------
def make_map_background():
    rnd = random.Random(7)                      # אותה "מפה" בכל הרצה
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60",
              "#2e6db4", "#8e44ad", "#2c2c2c", "#9e9e9e"]
    # ערים: נקודות פזורות על רשת, עם תזוזה אקראית קטנה
    cities = [(x * 150 + rnd.randint(20, 110), y * 130 + rnd.randint(20, 100))
              for x in range(8) for y in range(6)]
    parts = []
    # מסלולים: כל עיר מחוברת ל-2 הערים הקרובות אליה
    done = set()
    for i, (x1, y1) in enumerate(cities):
        nearest = sorted(range(len(cities)),
                         key=lambda j: (cities[j][0] - x1) ** 2 + (cities[j][1] - y1) ** 2)[1:3]
        for j in nearest:
            if (j, i) in done:
                continue
            done.add((i, j))
            x2, y2 = cities[j]
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="{rnd.choice(colors)}" stroke-width="9" '
                         f'stroke-dasharray="22 5" opacity="0.22"/>')
    for x, y in cities:
        parts.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#7a5230" '
                     f'stroke="#fffaf0" stroke-width="3" opacity="0.35"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="780">'
           f'{"".join(parts)}</svg>')
    return base64.b64encode(svg.encode()).decode()


# ---------- ממשק ----------
st.set_page_config(page_title="בוט טיקט טו רייד", page_icon="🚂")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@500;700&display=swap');

/* רקע בצבע קלף ישן, כיוון מימין לשמאל */
.stApp {
    background-color: #f3e6c8;
    background-image: url("data:image/svg+xml;base64,MAP_SVG");
    background-size: cover;
    background-attachment: fixed;
    direction: rtl;
    text-align: right;
    font-family: 'Frank Ruhl Libre', serif;
}
.stApp p, .stApp li { font-size: 1.1rem; color: #3b2a1a; }

/* כותרת בסגנון כרזת רכבת */
.ttr-title {
    font-family: 'Frank Ruhl Libre', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #7a2e1d;
    text-align: center;
    margin-bottom: 0.3rem;
}

/* פס בשמונת צבעי קלפי הקרון */
.ttr-stripe {
    height: 10px;
    border-radius: 5px;
    margin-bottom: 1.2rem;
    background: linear-gradient(to left,
        #c0392b 0 12.5%, #e67e22 12.5% 25%, #f1c40f 25% 37.5%, #27ae60 37.5% 50%,
        #2e6db4 50% 62.5%, #8e44ad 62.5% 75%, #2c2c2c 75% 87.5%, #fdfdfd 87.5% 100%);
    border: 1px solid #5a3b22;
}

/* בועת שאלה: כחול כהה */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #1f3a5f;
    border-radius: 14px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: #ffffff;
}

/* בועת תשובה: קרם עם מסגרת חומה, כמו כרטיס יעד */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #fffaf0;
    border: 2px solid #7a5230;
    border-radius: 14px;
}

/* כפתור "שיחה חדשה" בצבע קרון אדום-חום */
.stButton button {
    background: #9c3b26;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'Frank Ruhl Libre', serif;
}
.stButton button:hover { background: #7a2e1d; color: #ffffff; }
.stButton button p { color: #ffffff; }

/* הפס העליון שקוף, ואזור ההקלדה בצבע הרקע */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stBottom"] > div { background: rgba(243, 230, 200, 0.9); }
[data-testid="stChatInput"] > div { background: #fffaf0; border-color: #7a5230; }
</style>
""".replace("MAP_SVG", make_map_background()), unsafe_allow_html=True)

st.markdown('<div class="ttr-title">🚂 בוט חוקי טיקט טו רייד</div>'
            '<div class="ttr-stripe"></div>', unsafe_allow_html=True)

llm, query_engine = load_bot()

new_chat = st.button("שיחה חדשה")
if new_chat or "messages" not in st.session_state:
    st.session_state.messages = []   # מה שמוצג במסך
    st.session_state.history = []    # מה שנשלח לעיבוי

# הודעת פתיחה בכל שיחה
with st.chat_message("assistant"):
    st.write("היי, אני כאן כדי לעזור במשחק טיקט טו רייד! "
             "לכל שאלה שתתעורר לכם בכל שלב במשחק, אני כאן כדי לעזור.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

question = st.chat_input("שאלי שאלה על החוקים...")
if question:
    with st.chat_message("user"):
        st.write(question)
    answer = answer_question(question, st.session_state.history)
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages += [{"role": "user", "text": question},
                                  {"role": "assistant", "text": answer}]