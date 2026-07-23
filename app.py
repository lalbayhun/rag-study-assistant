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
def reformulate_query(query: str, model: str = "qwen2.5:7b") -> str:
    prompt = f"""Lütfen aşağıdaki sorudaki Türkçe yazım hatalarını düzelt.
Eğer bir hata yoksa soruyu aynen bırak. 
Cevap olarak sadece düzeltilmiş soruyu ver, hiçbir ek açıklama yapma.

Orijinal Soru: {query}
Düzeltilmiş Soru:"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        rephrased = response["message"]["content"].strip()
        rephrased = rephrased.replace("Düzeltilmiş Soru:", "").replace("Düzeltilmiş hali:", "").strip()
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
    prompt = f"""You are a study assistant for a course. Below are numbered text excerpts 
from lecture notes (in Turkish). Not all excerpts are necessarily relevant to the question.

Read the excerpts and follow exactly one of these two rules:

RULE A - If at least one excerpt contains information relevant to the question:
Answer the question in Turkish using only that information. 
Do not add any extra remarks about whether the information was found or not.
Do not mention "excerpt", "parça", or any numbering (like [Parça 1]) in your answer - 
just write the answer as if you knew it directly, in plain natural Turkish.

RULE B - If none of the excerpts contain information relevant to the question:
Respond with exactly this sentence and nothing else: "Bu bilgi verilen notlarda bulunmuyor."

Never combine both rules. Never write a real answer and then also say the information 
was not found (or vice versa).

Excerpts:
{context_text}

Question: {query}

Answer in Turkish:
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