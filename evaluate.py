import os, time, csv
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, PromptTemplate, QueryBundle
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=api_key)
llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=api_key)

collection = chromadb.PersistentClient(path="./ttr_db").get_collection("ticket_to_ride")
index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection),
                                           embed_model=embed_model)

prompt = PromptTemplate(
    "אתה עוזר שעונה על שאלות על חוקי המשחק טיקט טו רייד.\n"
    "כללים:\n"
    "1. ענה בעברית ובקצרה.\n"
    "2. ענה רק לפי הקטעים שלמטה. אם אין בהם תשובה, כתוב: אין לי מידע על זה.\n"
    "3. אם המשתמש לא ציין מפה, ענה לפי חוקי הבסיס.\n"
    "4. אם המהדורה לא מזכירה חוק מסוים, חוקי הבסיס חלים עליה.\n"
    "5. אל תציין את שם המפה בתשובה.\n\n"
    "קטעים:\n{context_str}\n\n"
    "שאלה: {query_str}\n"
    "תשובה:"
)
hyde = PromptTemplate("כתוב פסקה קצרה, כאילו היא מתוך חוברת החוקים של טיקט טו רייד, "
                      "שעונה על השאלה: {question}")
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3, text_qa_template=prompt)

tests = [
    ("כמה נקודות על מסלול באורך 6?", "15"),
    ("כמה קלפי קרון מקבל כל שחקן בהתחלה?", "4"),
    ("אפשר לקחת קטר גלוי כקלף השני?", "לא"),
    ("מתי המשחק נגמר?", "שחקן עם 2 קרונות או פחות, ואז תור אחרון לכולם"),
    ("כמה שווה בונוס המסלול הרציף הארוך?", "10 נקודות"),
    ("איך עובדות מנהרות באירופה?", "חושפים 3 קלפים, משלמים על כל תואם"),
    ("כמה עולה תחנת הרכבת השנייה באירופה?", "2 קלפים"),
    ("כמה שווה תחנה שלא נבנתה באירופה?", "4 נקודות"),
    ("כמה קרונות יש לכל שחקן בנורדי?", "40"),
    ("אפשר להשתמש בקטרים במסלול רגיל בנורדי?", "לא"),
    ("כמה נקודות מקבלים על רוב נוסעים בצבע בגרמניה?", "20"),
    ("כמה כרטיסים צריך להשלים כדי לנצח ב-First Journey?", "6"),
    ("ג'וקר?", "קטר מחליף כל צבע"),
    ("תיקו?", "הכי הרבה כרטיסים שהושלמו, ואז המסלול הארוך"),
    ("מה בירת צרפת?", "אין לי מידע"),
    ("איזו הרחבה עדיף לקנות לילדים?", "אין לי מידע"),
]

def ask(question, use_hyde):
    if use_hyde and len(question.split()) <= 4:
        guess = llm.complete(hyde.format(question=question)).text.strip()
        question = QueryBundle(query_str=question, custom_embedding_strs=[guess, question])
    return str(query_engine.query(question)).strip()

with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["#", "שאלה", "תשובה נכונה", "בלי HyDE", "עם HyDE", "נכון בלי?", "נכון עם?"])
    for i, (q, expected) in enumerate(tests, start=1):
        print(f"\n{i}. {q}\n   נכון: {expected}")
        plain = ask(q, use_hyde=False)
        print("   בלי HyDE:", plain)
        time.sleep(5)   # הפסקה כדי לא לחרוג ממגבלת הבקשות החינמית
        with_hyde = ask(q, use_hyde=True)
        print("   עם HyDE: ", with_hyde)
        time.sleep(5)
        writer.writerow([i, q, expected, plain, with_hyde, "", ""])

print("\nהתוצאות נשמרו ב-results.csv")