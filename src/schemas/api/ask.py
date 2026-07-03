from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: str = Field("llama3.2:1b", description="Ollama model to use for generation")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 3,
                "use_hybrid": True,
                "model": "llama3.2:1b",
                "categories": ["cs.AI", "cs.LG"],
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }


class AgenticAskResponse(AskResponse):
    """Response model for agentic RAG question answering."""

    reasoning_steps: List[str] = Field(..., description="Agent's decision-making steps")
    retrieval_attempts: int = Field(..., description="Number of document retrieval attempts")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for feedback and debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are neural network architectures...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "reasoning_steps": [
                    "Decided to retrieve relevant papers",
                    "Retrieved documents from database",
                    "Generated answer from relevant documents",
                ],
                "retrieval_attempts": 1,
                "trace_id": "abc123-def456-ghi789",
            }
        }


class SkepticReviewRequest(AskRequest):
    """Request model for skeptical research-paper review."""

    focus_area: Optional[str] = Field(
        None,
        description="Optional lens for the critique, such as methodology, evidence, or limitations",
        max_length=200,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Review the evidence behind transformer scaling laws",
                "focus_area": "limitations and unsupported claims",
                "top_k": 5,
                "use_hybrid": True,
                "model": "llama3.2:1b",
            }
        }


class SkepticReviewResponse(AgenticAskResponse):
    """Structured response for AI Research Paper Skeptic Agent output."""

    main_claim: str = Field(..., description="Concise statement of the paper/topic claim being evaluated")
    method: str = Field(..., description="Methods or experimental setup identified from retrieved evidence")
    evidence: List[str] = Field(..., description="Evidence points grounded in retrieved papers")
    limitations: List[str] = Field(..., description="Limitations, caveats, and threats to validity")
    unsupported_claims: List[str] = Field(..., description="Claims that need stronger evidence or were not supported by retrieval")
    questions_to_ask: List[str] = Field(..., description="Follow-up questions a reader should ask before trusting the claim")
    risk_score: int = Field(..., description="Skepticism risk score from 0 (low) to 100 (high)", ge=0, le=100)
    routing_decision: str = Field(..., description="Recommended next action for the user")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Review a paper about transformer scaling laws",
                "answer": "Structured skeptical review...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
                "chunks_used": 5,
                "search_mode": "hybrid",
                "reasoning_steps": ["Validated query scope", "Retrieved documents", "Applied unsupported-claim guardrail"],
                "retrieval_attempts": 1,
                "trace_id": None,
                "main_claim": "The paper argues that scaling improves model performance.",
                "method": "Retrieved evidence mentions empirical model comparisons.",
                "evidence": ["Relevant retrieved sources were used as evidence."],
                "limitations": ["Review depends on retrieved chunks, not a full peer review."],
                "unsupported_claims": ["Claims not present in retrieved evidence require verification."],
                "questions_to_ask": ["Are evaluation datasets representative?"],
                "risk_score": 45,
                "routing_decision": "Proceed with caution and inspect cited papers.",
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback on RAG answers."""

    trace_id: str = Field(..., description="Langfuse trace ID from the response")
    score: float = Field(..., description="Feedback score (0-1 or -1 to 1)", ge=-1, le=1)
    comment: Optional[str] = Field(None, description="Optional feedback comment", max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "abc123-def456-ghi789",
                "score": 1.0,
                "comment": "This answer was very helpful and accurate!",
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback recorded successfully",
            }
        }
