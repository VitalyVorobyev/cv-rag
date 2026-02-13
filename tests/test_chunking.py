from cv_rag.chunking import chunk_sections
from cv_rag.tei_extract import Section


def test_chunk_boundaries_and_overlap() -> None:
    text = " ".join(f"token{i:04d}" for i in range(900))
    sections = [Section(title="Methods", text=text)]

    chunks = chunk_sections(sections, max_chars=1200, overlap_chars=200)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 1200 for chunk in chunks)

    # Adjacent chunks should share vocabulary due to overlap.
    for prev, curr in zip(chunks, chunks[1:]):
        prev_tail = set(prev.text[-260:].split())
        curr_head = set(curr.text[:260].split())
        assert prev_tail.intersection(curr_head)
