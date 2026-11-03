# Policy Q&A Assistant (RAG)

A command-line assistant that answers employee questions about company policies. It retrieves the most relevant policy sections with vector search (Voyage AI embeddings) and has Claude answer using only those sections, citing the policy and section it relied on.

## How it works

1. **Load** every `.md` file in `policies/` and split it into sections at each `## ` heading.
2. **Embed** the sections with Voyage AI (`voyage-3-large`) and upsert them into a **Pinecone** serverless index (`policy-qa`, 1024 dimensions, cosine). This happens only the first time; later runs reuse the stored vectors.
3. **Retrieve** the top 4 sections closest to the user's question with a Pinecone query.
4. **Answer** with Claude (`claude-sonnet-5-5`), which is instructed to use only the retrieved sections, cite them, and say so (suggesting HR) when the policies don't cover the question.
5. **Remember** the conversation, so follow-up questions work.

## Project structure

```
RAGChatbot/
├── vectordbRag.py      # Pinecone indexing/retrieval, embedding helpers and the chat loop
├── policies/           # Policy documents (markdown)
│   ├── leave_policy.md
│   ├── remote_work_policy.md
│   ├── expense_policy.md
│   └── security_policy.md
├── .env                # API keys (not committed)
└── README.md
```

The sample policies are fictional. Replace or add your own `.md` files in `policies/`; they are picked up automatically.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install anthropic voyageai pinecone python-dotenv
   ```

2. Create a free Pinecone account at [pinecone.io](https://www.pinecone.io/) and copy an API key.

3. Create a `.env` file in the project root:

   ```
   ANTHROPIC_API_KEY=your-anthropic-key
   VOYAGE_API_KEY=your-voyage-key
   PINECONE_API_KEY=your-pinecone-key
   ```

The `policy-qa` index is created automatically on the first run (AWS `us-east-1`, serverless).

## Run

```bash
.venv/bin/python vectordbRag.py
```

To use a different policy folder:

```bash
.venv/bin/python vectordbRag.py ./my_policies
```

The first run embeds the policies and uploads them to Pinecone. Later runs skip that step. After you edit or add policy files, rebuild the index:

```bash
.venv/bin/python vectordbRag.py --reindex
```

Wait for `Ask a question, or type 'quit' to exit.`, then type a question at the `You:` prompt. Type `quit` to exit.

Example questions:

- How many PTO days can I carry over?
- Can I work from another country?
- What is the approval limit for a $2,000 expense?
- What is the hotel limit in major cities?

Each answer is followed by a `Retrieved:` line listing the sections that were passed to Claude, which is useful for checking retrieval quality.

## Writing policy documents

- Start each file with a single `# Title` line. It is used in citations.
- Use `## Section` headings. Each section becomes one searchable chunk, so keep each one focused on a single topic.

## Configuration

Constants near the bottom of `vectordbRag.py`:

| Name | Default | Purpose |
| --- | --- | --- |
| `CHAT_MODEL` | `claude-sonnet-5-5` | Claude model used to answer |
| `POLICY_DIR` | `./policies` | Folder of policy files |
| `TOP_K` | `4` | Number of sections retrieved per question |
| `INDEX_NAME` | `policy-qa` | Pinecone index name |
| `NAMESPACE` | `policies` | Pinecone namespace holding the vectors |

## Troubleshooting

- **`RateLimitError` from Voyage:** Accounts without a payment method are limited to 3 requests per minute. Indexing uses one request (first run or `--reindex`) and each question uses another. Wait a minute, or add a payment method at [dashboard.voyageai.com](https://dashboard.voyageai.com/) (the free token allowance still applies).
- **`ModuleNotFoundError`:** Run with `.venv/bin/python`, or select the `.venv` interpreter in VS Code (Cmd+Shift+P → "Python: Select Interpreter").
- **`No .md policy files found`:** Check that the policy folder exists and contains `.md` files.
- **`KeyError: 'PINECONE_API_KEY'`:** Add the key to `.env` (see Setup) and run from the project folder.
- **Missing API key errors:** Make sure `.env` is in the folder you run the script from.

## Notes

- Vectors persist in Pinecone, so changes to the files in `policies/` are not seen until you run with `--reindex`.
- Answers are only as good as the documents provided. For anything binding, confirm with HR.
