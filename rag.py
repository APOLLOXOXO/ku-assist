from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_MODEL, TOP_K_RESULTS, GROQ_API_KEY, GROQ_MODEL, TAVILY_API_KEY

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_prompt(mode):
    if mode == "understand":
        template = """You are a patient university tutor.
First explain the concept in simple plain language.
Then break down any equations symbol by symbol.
Use the context below from the student's actual notes.

Context: {context}
Question: {question}

Answer:"""

    elif mode == "exam":
        template = """You are helping a student prepare for an exam.
Return the EXACT wording from the context below. Do not paraphrase.
Do not add your own words. Quote directly from the notes.

Context: {context}
Question: {question}

Exact answer from notes:"""

    elif mode == "practical":
        template = """You are a practical engineering tutor.
Explain how this concept applies in the real world.
Give a concrete real-world example after explaining.
Use the context below from the student's notes.

Context: {context}
Question: {question}

Answer:"""

    else:
        template = """Answer the question using only the context below.
If the answer is not in the context say so honestly.

Context: {context}
Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def ask(question, mode="standard"):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )
    llm = OllamaLLM(model=OLLAMA_MODEL)
    prompt = get_prompt(mode)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": [doc.metadata for doc in docs]
    }

def ask_deep(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )
    docs = retriever.invoke(question)
    local_context = format_docs(docs)
    sources = [doc.metadata for doc in docs]

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL
    )

    evaluation_prompt = f"""You are an expert university tutor evaluating study notes.

A student asked: "{question}"

Here is what their lecture notes contain:
{local_context}

Identify specifically what is missing or incomplete in these notes
that would be needed for a complete answer.
List the gaps in one sentence each. Be specific.
If the notes are sufficient, say "SUFFICIENT"."""

    evaluation = llm.invoke([HumanMessage(content=evaluation_prompt)])
    gaps = evaluation.content

    web_context = ""
    web_sources = []
    if "SUFFICIENT" not in gaps:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)

        search_query = f"{question} university lecture notes"

        search_results = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "mit.edu", "stanford.edu", "ucl.ac.uk",
                "uonbi.ac.ke", "mak.ac.ug", "ug.edu.gh",
                "ieee.org", "researchgate.net"
            ]
        )

        for result in search_results["results"]:
            web_sources.append({
                "title": result["title"],
                "url": result["url"],
                "content": result["content"]
            })
            web_context += f"\nSource: {result['title']} ({result['url']})\n{result['content']}\n"

    synthesis_prompt = f"""You are an expert university tutor synthesizing information for a student.

Question: "{question}"

LOCAL NOTES CONTENT:
{local_context}

SUPPLEMENTARY INFORMATION:
{web_context if web_context else "Local notes were sufficient."}

Provide a structured answer with these exact sections:

## Architecture / Concept
Explain what this is in plain language first.

## Formula Breakdown
Show the formula and explain every single symbol.

## From Your Notes
Quote the most relevant parts from the local notes directly.

## Additional Context
Add anything from supplementary sources that deepens understanding.
Cite the exact URL and source name for each point.

## Summary
One paragraph pulling everything together."""

    final_response = llm.invoke([HumanMessage(content=synthesis_prompt)])

    return {
        "answer": final_response.content,
        "gaps_identified": gaps,
        "web_supplemented": "SUFFICIENT" not in gaps,
        "web_sources": web_sources,
        "sources": sources
    }