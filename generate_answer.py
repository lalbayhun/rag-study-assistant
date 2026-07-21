"""
Asama 3 & 4: Retrieval & Generation
Bu script:
1. embed_and_store.py'den ChromaDB baglantisini ve arama fonksiyonunu kullanir
2. Kullanicidan soru alir
3. Soruya en yakin chunk'lari bulur (retrieval)
4. Bu chunk'lari + soruyu birlestirip Ollama'ya gonderir (generation)
5. Ollama, sadece verilen baglama dayanarak cevap uretir
"""

import re

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Debug modu: True yaparsan, her soruda hangi chunk'larin bulundugunu ekrana basar.
# Sorun ararken True, normal kullanimda False yapabilirsin.
DEBUG = True


def load_existing_database(db_path: str = "./chroma_db", collection_name: str = "oop_notes"):
    """
    Daha once olusturulmus ChromaDB veritabanina baglanir.
    (embed_and_store.py zaten calistirilmis ve veritabani hazir olmali)
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return collection, embedding_model


def retrieve_relevant_chunks(query: str, collection, embedding_model, top_k: int = 3):
    """
    Soruyu vektore cevirir, veritabaninda en benzer chunk'lari bulup dondurur.
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results["documents"][0]


def build_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Bulunan chunk'lari ve kullanicinin sorusunu birlestirip,
    Ollama'ya gonderecegimiz prompt'u olusturur.

    Bu asama "prompt engineering" olarak adlandirilir - modele
    NASIL davranmasi gerektigini net bir sekilde soyluyoruz.
    """
    # Her chunk'i numaralandirarak veriyoruz, model hangi parcanin alakali oldugunu daha kolay ayirt etsin
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
    """
    Modelin cevabinda kalmis olabilecek "[Parca 1]", "Parca 2'deki" gibi
    ic sistem referanslarini temizler.

    Prompt talimatlari kucuk modellerde her zaman %100 uyulmuyor,
    bu yuzden ek bir guvenlik onlemi olarak metni de temizliyoruz.
    """
    # [Parca 1], [Parça 2] gibi koseli parantezli ifadeleri sil
    answer = re.sub(r"\[Par[çc]a\s*\d+\]", "", answer, flags=re.IGNORECASE)
    # "Parca 1'deki", "Parça 2 numaralı" gibi koseli parantezsiz ifadeleri sil
    answer = re.sub(r"Par[çc]a\s*\d+['’]?\w*\s*", "", answer, flags=re.IGNORECASE)
    # Temizlik sonrasi olusabilecek fazladan bosluklari duzelt
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer


def ask_local_llm(prompt: str, model: str = "qwen2.5:3b") -> str:
    """
    Olusturulan prompt'u Ollama'da calisan lokal LLM'e gonderir, cevabi dondurur.

    temperature: modelin ne kadar "yaratici/rastgele" davranacagini kontrol eder.
                 0'a yakin degerler daha tutarli/deterministik cevaplar verir,
                 RAG icin genellikle dusuk temperature tercih edilir.
    """
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

        # Retrieval: en alakali chunk'lari bul
        relevant_chunks = retrieve_relevant_chunks(user_question, collection, embedding_model)

        if DEBUG:
            print("\n[DEBUG] Bulunan chunk'lar:")
            for i, chunk in enumerate(relevant_chunks, start=1):
                print(f"--- Parca {i} ---")
                print(chunk[:200])

        # Generation: prompt olustur ve LLM'e gonder
        prompt = build_prompt(user_question, relevant_chunks)
        answer = ask_local_llm(prompt)
        answer = clean_answer(answer)  # guvenlik onlemi: sizan referanslari temizle

        print(f"\nCevap: {answer}\n")
        print("-" * 50)