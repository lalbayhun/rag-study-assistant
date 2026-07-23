"""
Streamlit Web Arayüzü (app.py)
Bu script:
1. Arka planda hazırladığımız RAG mantığını Streamlit bileşenleriyle birleştirir.
2. Kullanıcıya ChatGPT tarzı şık bir sohbet (chat) arayüzü sunar.
3. Soru yazma, analiz etme, kaynakları (chunks) gösterme ve yanıt üretme adımlarını yönetir.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import re
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sayfa Yapılandırması
st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- PDF İŞLEME VE CHUNKING YARDIMCILARI ---
def clean_noise(text: str) -> str:
    text = re.sub(r"(?im)^\s*Data\s*$", "", text)
    text = re.sub(r"(?im)^\s*Types\s*$", "", text)
    text = re.sub(r"(?im)^\s*Data\s*Types\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def process_uploaded_pdf(uploaded_file, db_path: str = "./chroma_db", collection_name: str = "oop_notes"):
    """Yüklenen PDF'i okur, temizler, chunk'lara böler ve ChromaDB'ye kaydeder."""
    # Geçici olarak dosyayı diske kaydedelim
    temp_pdf_path = f"./{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # PDF'i oku
    reader = PdfReader(temp_pdf_path)
    full_text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            cleaned = clean_noise(text)
            if cleaned:
                full_text += f"\n\n--- Page {page_number} ---\n{cleaned}"

    # Chunk'lara böl
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    # Embedding ve ChromaDB kayıt
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    embeddings = embedding_model.encode(chunks).tolist()
    base_name = uploaded_file.name.replace(".pdf", "")
    ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )

    # Geçici dosyayı temizle
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    return len(chunks)

# --- CACHE YÖNETİMİ ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_db_client():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(name="oop_notes")
    except Exception:
        collection = client.get_or_create_collection(name="oop_notes")
    return collection

embedding_model = load_embedding_model()
collection = load_db_client()

# --- YARDIMCI FONKSİYONLAR ---
#Çok dilli SEO promptu, Sıfır sıcaklık determinizmi, Hibrit Query Expansion ve temizlik katmani
def reformulate_query(query: str, model: str = "qwen2.5:7b") -> str:
    """
    Kullanıcı girdisini analiz eder:
    - Yazım hatalarını düzeltir.
    - Türkçe ve İngilizce akademik terimleri hibrit olarak genişleterek 
      (Query Expansion) veritabanı arama başarısını (recall) maksimize eder.
    """
    prompt = f"""You are an advanced bilingual academic search engine optimization (SEO) assistant. 
Your task is to take the user's question and expand it with both Turkish and English academic equivalents so it can match lecture notes in either language.
For example:
- If the user asks "tümevarım nedir", you should output: "tümevarım induction proof by induction"
- If the user asks "path nedir", you should output: "path yol graph"

Output ONLY the expanded search query. Do NOT add any extra explanation, quotation marks, or conversational filler.

Original Question: {query}
Expanded Query:"""
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        rephrased = response["message"]["content"].strip()
        
        for prefix in ["Expanded Query:", "Enriched Query:", "Optimized Query:", "Düzeltilmiş Soru:"]:
            rephrased = rephrased.replace(prefix, "")
        rephrased = rephrased.replace('"', '').strip()
        return rephrased
    except Exception:
        return query
    
def retrieve_relevant_chunks(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query]).tolist()
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        return results["documents"][0]
    except Exception:
        return []

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context_text = "\n\n".join(
        f"[Parca {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    
    prompt = f"""You are a strict, no-nonsense academic assistant. Answer the user's question using ONLY the provided lecture note excerpts.

CRITICAL RULES:
1. ABSOLUTE LANGUAGE PURITY: Detect the language of the user's question and write your ENTIRE response strictly in that language (Turkish or English). Never mix languages, never output Chinese, and never add conversational filler.
2. IF INFORMATION EXISTS: Write a direct, concise answer. At the very end of your answer, you MUST append the page citation using the exact format:
   - For Turkish questions: "(Kaynak: Sayfa [sayfa_numarasi])"
   - For English questions: "(Source: Page [page_number])"
   (Note: Extract the actual page number from the excerpts, never output letters like "X").
3. IF INFORMATION DOES NOT EXIST: If the answer is not in the excerpts, output ONLY the exact fallback sentence below, with NO page numbers:
   - For Turkish: "Bu bilgi verilen notlarda bulunmuyor."
   - For English: "This information is not available in the provided notes."

Excerpts:
{context_text}

Question: {query}

Answer:
"""
    return prompt

def clean_answer(answer: str) -> str:
    answer = re.sub(r"\[Par[çc]a\s*\d+\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"Par[çc]a\s*\d+['’]?\w*\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer

def ask_local_llm(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


# --- STREAMLIT UI (GELİŞTİRİLMİŞ ARAYÜZ) ---

st.title("🤖 RAG Study Assistant")
st.caption("Yerel LLM (Qwen2.5:7B) ve Dinamik Vektör Veritabanı Destekli Ders Asistanı")

# Sidebar (Kenar Çubuğu) - Dosya Yönetimi ve Ayarlar
with st.sidebar:
    st.header("📁 Ders Notu Yönetimi")
    uploaded_pdf = st.file_uploader("Yeni bir PDF ders notu yükle", type=["pdf"])
    
    if uploaded_pdf is not None:
        if st.button("Veritabanına İşle"):
            with st.spinner(f"'{uploaded_pdf.name}' okunuyor ve vektörleştiriliyor..."):
                try:
                    chunk_count = process_uploaded_pdf(uploaded_pdf)
                    st.success(f"Başarılı! {chunk_count} parça veritabanına eklendi.")
                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)}")

    st.markdown("---")
    st.header("⚙️ Sistem Ayarları")
    debug_mode = st.toggle("🔍 Debug Modunu Aç", value=False, help="Bulunan kaynak parçalarını (chunks) gösterir.")
    
    if st.button("🧹 Sohbet Geçmişini Temizle"):
        st.session_state.messages = []
        st.rerun()

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girdisi
if user_input := st.chat_input("Ders notlarıyla ilgili bir şeyler sor..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Notlar taranıyor ve yanıt üretiliyor..."):
            try:
                cleaned_question = reformulate_query(user_input)
                
                if debug_mode and cleaned_question.lower() != user_input.lower():
                    st.toast(f"Düzeltilen Soru: {cleaned_question}", icon="✍️")

                relevant_chunks = retrieve_relevant_chunks(cleaned_question, top_k=5)

                if debug_mode:
                    with st.expander("🔍 Bulunan Kaynak Parçalar (Debug)"):
                        if relevant_chunks:
                            for i, ch in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Parça {i}:**\n{ch}")
                        else:
                            st.warning("Veritabanında henüz hiç doküman bulunmuyor veya eşleşme yok.")

                prompt = build_prompt(cleaned_question, relevant_chunks)
                raw_answer = ask_local_llm(prompt)
                final_answer = clean_answer(raw_answer)

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                error_msg = f"Bir hata oluştu: {str(e)}. Lütfen Ollama'nın açık olduğundan emin olun."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})