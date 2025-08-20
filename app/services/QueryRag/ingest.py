import os
import asyncio
import chardet

from rag_core import split_text_iter, add_documents
from config import settings


def read_text_auto(path: str) -> str:
    # 讀 raw bytes
    raw = open(path, "rb").read()

    # 偵測編碼
    det = chardet.detect(raw)
    enc = det.get("encoding", "utf-8")
    print(f"[INGEST] {os.path.basename(path)} detected encoding = {enc}")

    # 嘗試多種編碼讀取
    for candidate in [enc, "utf-8-sig", "utf-8", "big5", "latin1"]:
        try:
            text = raw.decode(candidate, errors="ignore")
            # 偵測是否還是亂碼：如果全是問號或亂碼比例太高就跳過
            if text.count("�") / max(1, len(text)) < 0.05:
                # 順便轉成 UTF-8 覆蓋存檔，下次就乾淨了
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[INGEST] {os.path.basename(path)} converted to UTF-8.")
                return text
        except Exception:
            continue

    raise RuntimeError(f"Unable to decode {path} with reasonable encoding")


async def ingest_file(path: str):
    try:
        text = read_text_auto(path)

        # 切成 chunks
        chunks = []
        for i, piece in enumerate(split_text_iter(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
            chunks.append({"text": piece, "source": path, "chunk": i})

        if chunks:
            await add_documents(chunks)
            print(f"[INGEST] {os.path.basename(path)}: added {len(chunks)} chunks.")
        else:
            print(f"[INGEST] {os.path.basename(path)}: no valid chunks found.")

    except Exception as e:
        print(f"[ERROR] {path} failed: {e}")


async def ingest_dir(data_dir: str):
    files = []
    for root, _, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.endswith((".txt", ".md")):
                files.append(os.path.join(root, fn))

    print(f"[INGEST] Found {len(files)} file(s).")

    for i, path in enumerate(files, 1):
        print(f"[INGEST] ({i}/{len(files)}) {path}")
        await ingest_file(path)

    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(ingest_dir(settings.DATA_DIR))
