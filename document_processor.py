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


def read_pdf(pdf_path: str) -> str:
    """PDF'i okur ve tüm sayfaların metnini birleştirip döndürür."""
    reader = PdfReader(pdf_path)       # PdfReader nesnesi oluşturduk, pdf_path'de belirtilen dosyayı oluşturuyor.
    full_text = ""                     # çıkardığımız metinler buna eklenir

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            # Hangi sayfadan geldiğini işaretliyoruz, ileride kaynak göstermek için işimize yarayacak
            full_text += f"\n\n--- Page {page_number} ---\n{text}"

    return full_text


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
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


if __name__ == "__main__": #ana program olan dosya direkt çalıştırılıyorsa
    pdf_file = "OOPI_1.pdf"

    print("PDF okunuyor...")
    raw_text = read_pdf(pdf_file)
    print(f"Toplam karakter sayısı: {len(raw_text)}")

    print("\nMetin parçalara bölünüyor...")
    chunks = split_text_into_chunks(raw_text)
    print(f"Toplam chunk sayısı: {len(chunks)}")

    print("\n--- İLK 3 CHUNK ÖRNEĞİ ---")
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n[Chunk {i}] ({len(chunk)} karakter)")
        print(chunk)
        print("-" * 40)

    output_file = "chunks_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n\n===== CHUNK {i} =====\n{chunk}")

    print(f"\nTüm chunk'lar '{output_file}' dosyasına yazıldı.")