"""
Asama 3 & 4: Retrieval & Generation
Bu script:
1. embed_and_store.py'den ChromaDB baglantisini ve arama fonksiyonunu kullanir
2. Kullanicidan soru alir
3. Soruya en yakin chunk'lari bulur (retrieval)
4. Bu chunk'lari + soruyu birlestirip Ollama'ya gonderir (generation)
5. Ollama, sadece verilen baglama dayanarak cevap uretir
"""

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


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


def ask_local_llm(prompt: str, model: str = "qwen2.5:3b") -> str:
    """
    Olusturulan prompt'u Ollama'da calisan lokal LLM'e gonderir, cevabi dondurur.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
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

        # Generation: prompt olustur ve LLM'e gonder
        prompt = build_prompt(user_question, relevant_chunks)
        answer = ask_local_llm(prompt)

        print(f"\nCevap: {answer}\n")
        print("-" * 50) 