"""
Asama 2: Embedding & Storage
Bu script:
1. document_processor.py'den PDF okuma ve chunking fonksiyonlarini import eder
2. Her chunk'i bir embedding modeliyle vektore cevirir
3. Vektorleri ChromaDB'ye kaydeder (kalici, diskte saklanir)
4. Basit bir arama testi yapar, sistemin calistigini dogrular
"""

import chromadb
from sentence_transformers import SentenceTransformer
from document_processor import read_pdf, split_text_into_chunks


def build_vector_database(pdf_path: str, db_path: str = "./chroma_db", collection_name: str = "oop_notes"):
    """
    PDF'i okur, chunk'lara boler, embedding'e cevirir ve ChromaDB'ye kaydeder.
    """
    # 1. Adim: PDF'i oku ve chunk'lara bol
    print("PDF okunuyor ve parcalara bolunuyor...")
    raw_text = read_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)
    print(f"Toplam {len(chunks)} chunk olusturuldu.")

    # 2. Adim: Embedding modelini yukle
    # Bu model, cok dilli (Turkce dahil) calisir ve nispeten kucuk/hizlidir
    print("\nEmbedding modeli yukleniyor (ilk seferde internetten indirilecek)...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 3. Adim: ChromaDB istemcisini olustur (kalici, diske kaydeden versiyon)
    print("\nChromaDB baglantisi kuruluyor...")
    client = chromadb.PersistentClient(path=db_path)

    # Ayni isimde bir collection varsa once temizleyelim (tekrar calistirinca duplikasyon olmasin)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection zaten yoksa hata vermesi normal, gormezden gel

    collection = client.create_collection(name=collection_name)

    # 4. Adim: Her chunk icin embedding hesapla ve veritabanina ekle
    print("\nChunk'lar vektore cevriliyor ve veritabanina ekleniyor...")
    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"\nBasarili! {len(chunks)} chunk '{db_path}' klasorune kaydedildi.")
    return collection, embedding_model


def search_similar_chunks(query: str, collection, embedding_model, top_k: int = 3):
    """
    Bir soruyu vektore cevirir, veritabaninda en benzer chunk'lari bulur.
    """
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    return results["documents"][0]  # en yakin bulunan metinler


if __name__ == "__main__":
    pdf_file = "OOPI_1.pdf"

    collection, embedding_model = build_vector_database(pdf_file)

    # Test: static degiskenler hakkinda soru soralim
    test_query = "static degisken nedir"
    print(f"\n\n--- TEST SORGUSU: '{test_query}' ---")
    results = search_similar_chunks(test_query, collection, embedding_model)

    for i, result in enumerate(results, start=1):
        print(f"\n[Sonuc {i}]")
        print(result[:300])  # ilk 300 karakteri goster
        print("-" * 40)
