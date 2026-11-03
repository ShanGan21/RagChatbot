# %pip install voyageai

# Client Setup
from importlib import import_module
import voyageai

try:
    load_dotenv = import_module("dotenv").load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda: None

load_dotenv()

client = voyageai.Client()

# Chunk by section
import re


def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)

# Embedding Generation
def generate_embedding(chunks, model="voyage-3-large", input_type="query"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]
    result = client.embed(input, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]

# Pinecone vector store
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "policy-qa"
NAMESPACE = "policies"
EMBEDDING_DIM = 1024  # voyage-3-large default output dimension

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_index(reset: bool = False):
    """Return the Pinecone index, creating it first if it doesn't exist."""
    if reset and pc.has_index(INDEX_NAME):
        pc.delete_index(INDEX_NAME)

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def index_is_empty(index) -> bool:
    stats = index.describe_index_stats()
    return stats.namespaces.get(NAMESPACE) is None or stats.namespaces[NAMESPACE].vector_count == 0


# Policy Q&A Assistant (RAG)
CHAT_MODEL = "claude-sonnet-5-5"
POLICY_DIR = Path("./policies")
TOP_K = 4

llm = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are a company Policy Q&A Assistant. Answer employee questions using ONLY the "
    "policy excerpts provided in <context>. Be concise and specific, including numbers, "
    "limits and deadlines. Cite the policy and section you relied on, e.g. "
    "(Leave Policy > PTO Carryover). If the excerpts do not answer the question, say you "
    "couldn't find it in the policies and suggest contacting HR. Never guess or invent policy."
)


def load_policy_chunks(policy_dir: Path) -> List[Dict[str, str]]:
    """Split each policy file by section, tagging every chunk with its source."""
    files = sorted(policy_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md policy files found in {policy_dir.resolve()}")

    chunks = []
    for path in files:
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        for section in chunk_by_section(text)[1:]:  # [0] is just the document title
            heading, _, body = section.partition("\n")
            chunks.append(
                {
                    "source": f"{title} > {heading.strip()}",
                    # Prefix with source so the embedding captures which policy it belongs to
                    "content": f"{title} > {heading.strip()}\n{body.strip()}",
                }
            )
    return chunks


def index_policies(index, policy_dir: Path) -> int:
    chunks = load_policy_chunks(policy_dir)
    # Embed everything in one bulk call to avoid rate limiting errors from VoyageAI
    embeddings = generate_embedding([c["content"] for c in chunks], input_type="document")

    index.upsert(
        vectors=[
            {"id": f"policy-{i}", "values": emb, "metadata": chunk}
            for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
        ],
        namespace=NAMESPACE,
    )
    return len(chunks)


def retrieve(index, question: str) -> List[Dict[str, Any]]:
    result = index.query(
        vector=generate_embedding(question),
        top_k=TOP_K,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return [m.metadata for m in result.matches]


def answer(index, history: List[Dict[str, str]], question: str) -> Tuple[str, List[str]]:
    docs = retrieve(index, question)
    context = "\n\n---\n\n".join(d["content"] for d in docs)

    # Only the current turn carries retrieved context; history keeps plain questions/answers
    prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
    response = llm.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=history + [{"role": "user", "content": prompt}],
    )
    reply = "".join(b.text for b in response.content if b.type == "text")

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})
    return reply, [d["source"] for d in docs]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reindex = "--reindex" in sys.argv
    policy_dir = Path(args[0]) if args else POLICY_DIR

    index = get_index(reset=reindex)
    if reindex or index_is_empty(index):
        print(f"Indexing policies in {policy_dir} into Pinecone...")
        count = index_policies(index, policy_dir)
        print(f"Indexed {count} policy sections.")
    else:
        print("Using existing Pinecone index (run with --reindex to rebuild).")
    print("Ask a question, or type 'quit' to exit.\n")

    history: List[Dict[str, str]] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        reply, sources = answer(index, history, question)
        print(f"\nAssistant: {reply}")
        print("\nRetrieved: " + "; ".join(sources) + "\n")


if __name__ == "__main__":
    main()
