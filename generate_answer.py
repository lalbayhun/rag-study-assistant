"""
Asama 3 & 4: Retrieval & Generation
Bu script:
1. embed_and_store.py'den ChromaDB baglantisini ve arama fonksiyonunu kullanir
2. Kullanicidan soru alir
3. Soruyu olası yazım hatalarına karşı LLM ile düzeltir (Reformulation)
4. Soruya en yakin chunk'lari bulur (retrieval)
5. Bu chunk'lari + soruyu birlestirip Ollama'ya gonderir (generation)
6. Ollama, sadece verilen baglama dayanarak cevap uretir
"""

import re
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Debug modu: True yaparsan, her soruda hangi chunk'larin bulundugunu ekrana basar.
DEBUG = True

def load_existing_database(db_path: str = "./chroma_db", collection_name: str = "oop_notes"):
    """Daha once olusturulmus ChromaDB veritabanina baglanir."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return collection, embedding_model

def retrieve_relevant_chunks(query: str, collection, embedding_model, top_k: int = 5):
    """
    Soruyu vektore cevirir, veritabaninda en benzer chunk'lari bulup dondurur.
    top_k = 5 yapıldı: Yazım hataları veya eksik eşleşmelere karşı arama ağı genişletildi.
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results["documents"][0]

def build_prompt(query: str, context_chunks: list[str]) -> str:
    """Ollama'ya gonderecegimiz prompt'u olusturur."""
    context_text = "\n\n".join(
        f"[Parca {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are a study assistant for a course. Below are numbered text excerpts 
from lecture notes (in Turkish). Not all excerpts are necessarily relevant to the question.

Read the excerpts and follow exactly one of these two rules:

RULE A - If at least one excerpt contains information relevant to the question:
Answer the question in Turkish using only that information. 
Do not add any extra remarks about whether the information was found or not.
Do not mention "excerpt", "parça", or any numbering (like [Parça 1]) in your answer.

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
    """Modelin cevabinda kalmis olabilecek ic sistem referanslarini temizler."""
    answer = re.sub(r"\[Par[çc]a\s*\d+\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"Par[çc]a\s*\d+['’]?\w*\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer

def reformulate_query(query: str, model: str = "qwen2.5:7b") -> str:
    """
    Kullanicinin girdigi sorudaki yazim hatalarini (typo) duzeltmek ve 
    soruyu netlestirmek icin LLM'i kullanir. 
    """
    prompt = f"""Lütfen aşağıdaki sorudaki Türkçe yazım hatalarını düzelt ve anlamsız harfleri toparla.
Eğer bir hata yoksa soruyu aynen bırak. 
Cevap olarak sadece düzeltilmiş soruyu ver, hiçbir ek açıklama yapma.

Orijinal Soru: {query}
Düzeltilmiş Soru:"""
    
    # Hızlı ve tutarlı bir yanıt almak için temperature 0.0 kullanıyoruz
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    
    rephrased = response["message"]["content"].strip()
    rephrased = rephrased.replace("Düzeltilmiş Soru:", "").replace("Düzeltilmiş hali:", "").replace('"', '').strip()
    return rephrased

def ask_local_llm(prompt: str, model: str = "qwen2.5:7b") -> str:
    """Prompt'u Ollama'da calisan lokal LLM'e gonderir, cevabi dondurur."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print("Veritabanina baglaniliyor...")
    collection, embedding_model = load_existing_database()

    print("Hazir! Sorularini yazabilirsin. Cikmak icin 'q' yaz.\n")

    while True:
        user_question = input("Sorunuz: ")

        if user_question.lower() == "q":
            print("Gorusmek uzere!")
            break

        # YENI EKLENEN KISIM: Soruyu Düzeltme (Query Reformulation)
        print("\nSoru analiz ediliyor...")
        clean_question = reformulate_query(user_question)
        
        if DEBUG and clean_question.lower() != user_question.lower():
            print(f"[DEBUG] Soru duzeltildi: '{clean_question}'")

        # DEGISTIRILEN KISIM: Artik 'user_question' yerine 'clean_question' gonderiyoruz
        # Retrieval: en alakali chunk'lari bul
        relevant_chunks = retrieve_relevant_chunks(clean_question, collection, embedding_model)

        if DEBUG:
            print("\n[DEBUG] Bulunan chunk'lar:")
            for i, chunk in enumerate(relevant_chunks, start=1):
                print(f"--- Parca {i} ---")
                print(chunk[:200])

        # DEGISTIRILEN KISIM: Prompt'a da 'clean_question' gonderiyoruz
        # Generation: prompt olustur ve LLM'e gonder
        prompt = build_prompt(clean_question, relevant_chunks)
        answer = ask_local_llm(prompt)
        answer = clean_answer(answer)  # guvenlik onlemi: sizan referanslari temizle

        print(f"\nCevap: {answer}\n")
        print("-" * 50)