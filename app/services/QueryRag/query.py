import asyncio
import sys
from rag_core import rag_answer

async def main():
    query = " ".join(sys.argv[1:]).strip()
    print(f"[QUERY] {query}")
    result = await rag_answer(query)

    print("\n=== 答案 ===")
    print(result["answer"])
    print("\n=== 來源 ===")
    for s in result["sources"]:
        print(f"- {s['metadata']} : {s['text'][:80]}...")

if __name__ == "__main__":
    asyncio.run(main())
