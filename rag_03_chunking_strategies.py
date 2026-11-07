# rag_03_chunking_strategies.py — Splitting real documents intelligently

from pypdf import PdfReader

# ── 1. LOAD THE PDF ───────────────────────────────────────────────────────
print("Loading PDF...")
reader = PdfReader("company_doc.pdf")

# Extract all text from every page
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# PdfReader opens the file. Then we loop through every page and extract its text, 
# gluing it all together into one giant string called full_text. 
# After this line, the entire PDF is just a Python string — nothing special, same as if you typed it yourself.    

print(f"Total characters extracted: {len(full_text)}")
print(f"Total pages: {len(reader.pages)}")
print(f"\nFirst 500 characters preview:")
print(full_text[:500])
print("\n" + "─"*60)

# ── 2. STRATEGY 1 — FIXED SIZE CHUNKING ──────────────────────────────────


def fixed_size_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of fixed character size.
    overlap = how many characters are shared between consecutive chunks
    so we don't lose meaning at the boundaries.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # step back by overlap before next chunk

    return chunks

# Imagine cutting a long ribbon into pieces of exactly 500 characters. 
# The overlap=50 means before cutting, you step back 50 characters — 
# so consecutive chunks share 50 characters at their boundary.
# Overlap ensures that boundary sentence appears in both chunks.

fixed_chunks = fixed_size_chunks(full_text)
print(f"\nStrategy 1 — Fixed size chunking:")
print(f"  Total chunks created: {len(fixed_chunks)}")
print(
    f"  Average chunk size: {sum(len(c) for c in fixed_chunks) // len(fixed_chunks)} chars")
print(f"\n  Example chunk #5:")
print(f"  {fixed_chunks[min(5, len(fixed_chunks)-1)]}")
print("\n" + "─"*60)

# ── 3. STRATEGY 2 — PARAGRAPH CHUNKING ───────────────────────────────────


def paragraph_chunks(text):
    """
    Split on double newlines (natural paragraph boundaries).
    More meaningful than fixed size — respects document structure.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Filter out very short paragraphs (likely headers or noise)
    paragraphs = [p for p in paragraphs if len(p) > 100]
    return paragraphs

# Instead of cutting at fixed character counts, we split on \n\n — 
# double newlines, which is how paragraphs are separated in plain text. 
# Then we filter out anything shorter than 100 characters to remove headers, page numbers, and noise.

para_chunks = paragraph_chunks(full_text)
print(f"\nStrategy 2 — Paragraph chunking:")
print(f"  Total chunks created: {len(para_chunks)}")
print(
    f"  Average chunk size: {sum(len(c) for c in para_chunks) // len(para_chunks)} chars")
print(f"\n  Example chunk #3:")
print(f"  {para_chunks[min(3, len(para_chunks)-1)]}")
print("\n" + "─"*60)

# ── 4. STRATEGY 3 — SLIDING WINDOW CHUNKING ──────────────────────────────


def sliding_window_chunks(text, chunk_size=300, step=150):
    """
    Move a window across the text, stepping by half the chunk size.
    Higher overlap than fixed chunking — better for dense technical docs.
    step = how far we move forward each time (step < chunk_size = overlap)
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # skip tiny leftover chunks
            break
        chunks.append(" ".join(chunk_words))

    return chunks


sliding_chunks = sliding_window_chunks(full_text)
print(f"\nStrategy 3 — Sliding window chunking:")
print(f"  Total chunks created: {len(sliding_chunks)}")
print(
    f"  Average chunk size: {sum(len(c) for c in sliding_chunks) // len(sliding_chunks)} chars")
print(f"\n  Example chunk #5:")
print(f"  {sliding_chunks[min(5, len(sliding_chunks)-1)]}")
print("\n" + "─"*60)

# ── 5. COMPARISON SUMMARY ─────────────────────────────────────────────────
print("\nSummary:")
print(f"  Fixed size:     {len(fixed_chunks):>4} chunks")
print(f"  Paragraph:      {len(para_chunks):>4} chunks")
print(f"  Sliding window: {len(sliding_chunks):>4} chunks")
print("\nNext step: embed these chunks and compare retrieval quality.")
