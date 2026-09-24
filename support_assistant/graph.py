from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from support_assistant.prompts import build_prompt
from support_assistant.schemas import AnswerOutput


CHROMA_DIR = "support_assistant/chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name="zepto_policies")


class State(TypedDict):
    query: str
    intent: str
    context: str
    prompt: str
    answer: str
    sources: list[str]
    confidence: float
    retry_count: int
    validation_passed: bool


def classify_intent(state: State):
    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "giftcard",
        "support",
        "customer support",
    ]

    if any(word in query for word in keywords):
        return {"intent": "policy_question"}

    return {"intent": "general_question"}


def retrieve_context(state: State):
    query = state["query"]

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    context = documents[0]

    prompt = build_prompt(
        query=query,
        context=context
    )

    return {
        "context": context,
        "prompt": prompt,
        "sources": [ids[0]]
    }


def generate_answer(state: State):
    query = state["query"].lower()
    context = state["context"].lower()

    answer = ""

    if "delivery fee" in query:
        if "inr 149" in context and "inr 25" in context:
            answer = (
                "Standard delivery is free on orders over INR 149. "
                "Orders below INR 149 have a flat INR 25 delivery fee. "
                "Priority delivery costs an additional INR 15."
            )

    elif (
        ("return" in query or "refund" in query)
        and (
            "damaged" in query
            or "spoiled" in query
            or "incorrect" in query
        )
    ):
        answer = (
            "Grocery and perishable items may be reported for a return "
            "within 24 hours of delivery if they are damaged, spoiled, "
            "or incorrect. Return pickup, where required, is free of cost."
        )

    elif "cancel" in query or "cancellation" in query:
        answer = (
            "Orders can be cancelled free of cost before the order status "
            "changes to 'Packed'. Once an order has been packed, it cannot "
            "be cancelled through the app."
        )

    elif "gift card" in query or "giftcard" in query:
        answer = (
            "Zepto gift cards are available in INR 100, INR 250, "
            "INR 500, and INR 1000 denominations. They are valid "
            "for 1 year from the date of issue."
        )

    elif "support" in query:
        answer = (
            "Zepto customer support is available through in-app chat "
            "24 hours a day, 7 days a week. Email support is also "
            "available for non-urgent queries."
        )

    elif "track" in query or "tracking" in query:
        answer = (
            "Every Zepto order shows a live rider-tracking map from "
            "the moment it is packed until delivery. The estimated "
            "delivery time updates automatically as the rider moves."
        )

    elif "membership" in query or "pass" in query:
        answer = (
            "Zepto offers three account tiers: Basic, Zepto Pass, "
            "and Zepto Pass+. Membership can be cancelled at any time "
            "from account settings."
        )

    else:
        answer = (
            "Based on the retrieved Zepto policy information:\n\n"
            + state["context"]
        )

    return {
        "answer": answer,
        "confidence": 0.90
    }


def validate_output(state: State):
    try:
        AnswerOutput(
            answer=state["answer"],
            sources=state["sources"],
            confidence=state["confidence"]
        )

        # IMPORTANT:
        # Successful validation must explicitly mark True.
        return {
            "validation_passed": True
        }

    except Exception:
        # Retry only a maximum of 2 times.
        return {
            "validation_passed": False,
            "retry_count": state["retry_count"] + 1
        }


def check_validation(state: State):
    if state["validation_passed"]:
        return "finish"

    if state["retry_count"] < 2:
        return "retry"

    return "finish"


def direct_answer(state: State):
    output = AnswerOutput(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )

    return {
        "answer": output.answer,
        "sources": output.sources,
        "confidence": output.confidence,
        "validation_passed": True
    }


def route_question(state: State):
    if state["intent"] == "policy_question":
        return "retrieve"

    return "direct"


# -----------------------------
# BUILD LANGGRAPH
# -----------------------------

builder = StateGraph(State)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_answer", generate_answer)
builder.add_node("validate_output", validate_output)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(
    START,
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve": "retrieve_context",
        "direct": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_context",
    "generate_answer"
)

builder.add_edge(
    "generate_answer",
    "validate_output"
)

builder.add_conditional_edges(
    "validate_output",
    check_validation,
    {
        "retry": "generate_answer",
        "finish": END
    }
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()


def ask_question(query: str):
    result = graph.invoke(
        {
            "query": query,
            "intent": "",
            "context": "",
            "prompt": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "retry_count": 0,
            "validation_passed": False
        },
        config={
            "recursion_limit": 20
        }
    )

    return result


if __name__ == "__main__":
    question = input("Enter your question: ")

    result = ask_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConfidence:")
    print(result["confidence"])