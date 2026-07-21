"""
Aşama 1: Document Parsing & Chunking
Bu script:
1. PDF dosyasını okur
2. Sayfa sayfa metni çıkarır
3. Metni küçük, mantıklı parçalara (chunk) böler
4. Sonucu kontrol edebilmen için bir .txt dosyasına yazar
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Bazi sayfalarda dekoratif bir akis semasindan (Data -> Types -> Data Types...)
# kalma, anlamsizca tekrarlayan tek kelimelik satirlar cikiyor. Bunlar gercek
# icerik degil, embedding'i ve modelin anlayisini bozan gurultu - temizleyecegiz.
NOISE_LINES = {"data", "types", "data types"}


def clean_page_text(text: str) -> str:
    """
    Sayfa metninde, tek basina duran ve anlam tasimayan gurultu satirlarini
    (orn. dekoratif 'Data' / 'Types' kutucuklarindan kalma satirlar) siler.
    """
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if line.strip().lower() not in NOISE_LINES]
    return "\n".join(cleaned_lines)


def read_pdf(pdf_path: str) -> str:
    """PDF'i okur ve tüm sayfaların metnini birleştirip döndürür."""
    reader = PdfReader(pdf_path)
    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            text = clean_page_text(text)  # gurultuyu temizle
            # Hangi sayfadan geldiğini işaretliyoruz, ileride kaynak göstermek için işimize yarayacak
            full_text += f"\n\n--- Page {page_number} ---\n{text}"

    return full_text


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 150):
    """
    Metni küçük parçalara (chunk) böler.

    chunk_size: her parçanın yaklaşık karakter sayısı
    chunk_overlap: parçalar arası kaç karakterlik ortak bölge olsun
                   (cümle/anlam bütünlüğü kaybolmasın diye)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # önce paragraf, sonra satır, sonra cümle böler
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    pdf_file = "OOPI_1.pdf"

    print("PDF okunuyor...")
    raw_text = read_pdf(pdf_file)
    print(f"Toplam karakter sayısı: {len(raw_text)}")

    print("\nMetin parçalara bölünüyor...")
    chunks = split_text_into_chunks(raw_text)
    print(f"Toplam chunk sayısı: {len(chunks)}")

    # Kontrol edebilmen için ilk 3 chunk'ı ekrana basalım
    print("\n--- İLK 3 CHUNK ÖRNEĞİ ---")
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n[Chunk {i}] ({len(chunk)} karakter)")
        print(chunk)
        print("-" * 40)

    # Tüm chunk'ları bir dosyaya yazalım, gözden geçirmen için
    output_file = "chunks_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n\n===== CHUNK {i} =====\n{chunk}")

    print(f"\nTüm chunk'lar '{output_file}' dosyasına yazıldı.")