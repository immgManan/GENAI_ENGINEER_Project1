# This handles 1. chroma 2. product filtering 3. sematic search 4. LLM 5. RAG Prompt

from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing.")

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_core.prompts import (
    ChatPromptTemplate
)


# ==================================================
# CONFIGURATION
# ==================================================

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "customer_support_tickets"


# ==================================================
# LOAD CHROMA DATABASE
# ==================================================

def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    vectorstore = Chroma(

        collection_name=COLLECTION_NAME,

        embedding_function=embeddings,

        persist_directory=CHROMA_PATH
    )


    return vectorstore


# ==================================================
# RETRIEVE RELEVANT TICKETS
# ==================================================

def get_relevant_tickets(
    vectorstore,
    question,
    product,
    k=5
):

    """
    Retrieve tickets using:

    1. Product metadata filter
    2. Semantic similarity
    """


    documents = vectorstore.similarity_search(

        query=question,

        k=k,

        filter={
            "product": product
        }
    )


    return documents


# ==================================================
# GENERATE RAG ANSWER
# ==================================================

def generate_answer(
    question,
    product,
    documents
):

    if not documents:

        return None


    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    context_parts = []


    for index, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(

            f"""
Historical Ticket {index}

{document.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ==================================================
    # RAG PROMPT
    # ==================================================

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Customer Support Copilot.

Your job is to help a customer-support agent
solve a customer's problem using historical
customer-support tickets.

Selected Product:
{product}


Customer Question:
{question}


Historical Support Tickets:
{context}


IMPORTANT RULES:

1. Use the historical tickets as the primary
   source of information.

2. Do not invent historical tickets.

3. Do not invent resolutions.

4. Do not claim that a solution was previously
   used unless it appears in the provided tickets.

5. If the historical information is insufficient,
   clearly say that there is insufficient
   historical information.

6. You may provide a reasonable troubleshooting
   recommendation, but clearly distinguish it
   from historical evidence.

7. Never expose customer names, email addresses,
   or other personally identifiable information.

8. Keep the answer professional and useful.


Return the response using exactly these sections:


### Recommended Action

Provide the recommended action for the support agent.


### Historical Evidence

Explain what the relevant historical tickets
indicate.


### Suggested Customer Response

Write a professional response that the support
agent could send to the customer.
"""
    )


    # ==================================================
    # FORMAT PROMPT
    # ==================================================

    formatted_prompt = prompt.format(

        product=product,

        question=question,

        context=context
    )


    # ==================================================
    # LLM
    # ==================================================

    llm = ChatOpenAI(

        model="gpt-4.1-mini",

        temperature=0
    )


    # ==================================================
    # GENERATE RESPONSE
    # ==================================================

    response = llm.invoke(
        formatted_prompt
    )


    return response.content