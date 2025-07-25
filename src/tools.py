import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

import json 
import bs4

f = open("./data/general_guide.jsonl", 'r')
examples = []
for line in f:
    examples.append(json.loads(line))

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatIP(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=True,
)

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Create vector store for blogs 
def extract_title(html_text: str) -> str:
    """Extracts the <title> tag from an HTML document string."""
    soup = bs4.BeautifulSoup(html_text, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fallback: try first <h1> tag
    h1 = soup.find("h1")
    if h1 and h1.string:
        return h1.string.strip()
    return "Untitled Blog"

blog_urls = [
    "https://pacagen.com/blogs/cat-allergies/cat-adoption-guide",
    "https://pacagen.com/blogs/dust-allergies/product-review-guide-dust-mite-allergy-bedding-and-covers",
]

blog_loader = WebBaseLoader(web_paths=blog_urls)

blog_docs = blog_loader.load()
blog_splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(blog_docs)

blog_vector_store = FAISS.from_documents(blog_splits, embedding=embeddings)
for i, doc in enumerate(blog_docs):
    doc.metadata["source"] = blog_urls[i]
    doc.metadata["title"] = extract_title(doc.page_content)  # optional
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Create question vector store for blogs
docs = []
for ex in examples:
    content = "\n".join([f"{'question' if msg['role'] == 'user' else 'answer'}: {msg['content'].strip()}" for msg in ex["messages"]])
    docs.append(Document(page_content=content))

_ = vector_store.add_documents(documents=docs)
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Def tools 
# RAG for questions 
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search_with_score(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}\nScore:{score}")
        for doc, score in retrieved_docs
    )
    docs = []
    scores = []
    for doc, score in retrieved_docs:
        docs.append(doc)
        scores.append(score)
    return serialized, docs



# recommend blogs
@tool
def recommend_blog(query: str) -> str:
    """Recommend blog posts related to the query."""
    results = blog_vector_store.similarity_search_with_score(query, k=3)
    response = "Here are some blogs you might find helpful:<br><br>"
    for doc, score in results:
        title = doc.metadata.get("title", "Untitled Blog")
        url = doc.metadata.get("source", "#")
        response += f"<b>{title}</b><br><a href='{url}' target='_blank'>{url}</a><br><br>"
    return response

# ------------------------------------------------------------------------------------------------------------------------------------------------
