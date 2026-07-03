import json
import logging
from typing import Any, Iterator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG"]


def _parse_categories(categories: str) -> list[str] | None:
    """Parse a comma-separated category string into an API payload value."""
    return [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else None


def _format_source(source: Any) -> str:
    """Format string or dictionary source objects for Gradio markdown."""
    if isinstance(source, str):
        label = source.split("/")[-1]
        return f"[{label}]({source})"

    if isinstance(source, dict):
        title = source.get("title") or source.get("arxiv_id") or source.get("url") or "Source"
        url = source.get("url")
        if url:
            return f"[{title}]({url})"
        return str(title)

    return str(source)


def _format_skeptic_review(data: dict[str, Any]) -> str:
    """Convert skeptic-review API JSON into readable Gradio markdown."""
    evidence = data.get("evidence") or []
    limitations = data.get("limitations") or []
    unsupported_claims = data.get("unsupported_claims") or []
    questions = data.get("questions_to_ask") or []
    reasoning_steps = data.get("reasoning_steps") or []
    sources = data.get("sources") or []

    def bullets(items: list[Any]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- Not provided by the API response."

    markdown = f"""# 🕵️ Skeptic Review

## Main Claim
{data.get("main_claim", "Not provided")}

## Method / Evidence Base
{data.get("method", "Not provided")}

## Skeptical Answer
{data.get("answer", "No answer returned.")}

## Evidence
{bullets(evidence)}

## Limitations
{bullets(limitations)}

## Unsupported or Weakly Supported Claims
{bullets(unsupported_claims)}

## Questions to Ask
{bullets(questions)}

## Risk & Routing
- **Risk score:** {data.get("risk_score", "unknown")}/100
- **Routing decision:** {data.get("routing_decision", "Not provided")}

## Retrieval Info
- **Search mode:** {data.get("search_mode", "unknown")}
- **Chunks used:** {data.get("chunks_used", 0)}
- **Retrieval attempts:** {data.get("retrieval_attempts", 0)}
"""

    if sources:
        markdown += f"\n## Sources ({len(sources)})\n"
        for index, source in enumerate(sources[:5], 1):
            markdown += f"{index}. {_format_source(source)}\n"
        if len(sources) > 5:
            markdown += f"- ... and {len(sources) - 5} more\n"

    if reasoning_steps:
        markdown += "\n## Reasoning Steps\n"
        markdown += bullets(reasoning_steps)

    return markdown


async def stream_response(
    query: str, top_k: int = 3, use_hybrid: bool = True, model: str = DEFAULT_MODEL, categories: str = ""
) -> Iterator[str]:
    """Stream response from the RAG API"""
    if not query.strip():
        yield "Please enter a question."
        return

    # Parse categories
    category_list = _parse_categories(categories)

    # Prepare request payload
    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid, "model": model, "categories": category_list}

    try:
        url = f"{API_BASE_URL}/stream"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            data = json.loads(data_str)

                            # Handle error
                            if "error" in data:
                                yield f"Error: {data['error']}"
                                return

                            # Handle metadata
                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                continue

                            # Handle streaming chunks
                            if "chunk" in data:
                                current_answer += data["chunk"]
                                # Format response with sources if we have them
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                                            formatted_response += f"  {i}. {_format_source(source)}\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response

                            # Handle completion
                            if data.get("done", False):
                                final_answer = data.get("answer", current_answer)
                                if final_answer != current_answer:
                                    current_answer = final_answer

                                # Final formatted response
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):
                                            formatted_response += f"  {i}. {_format_source(source)}\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response
                                break

                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON lines

    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"


async def skeptic_review_response(
    query: str,
    focus_area: str = "",
    top_k: int = 5,
    use_hybrid: bool = True,
    model: str = DEFAULT_MODEL,
    categories: str = "",
) -> str:
    """Fetch a structured skeptical review from the Agentic RAG API."""
    if not query.strip():
        return "Please enter a research paper, claim, or topic to review."

    payload = {
        "query": query,
        "focus_area": focus_area.strip() or None,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "categories": _parse_categories(categories),
    }

    try:
        url = f"{API_BASE_URL}/skeptic-review"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code}\n\n{response.text}"

        return _format_skeptic_review(response.json())

    except httpx.RequestError as e:
        return f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def create_gradio_interface():
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="arXiv Paper Curator - RAG Chat",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # 🔬 arXiv Paper Curator - RAG Chat
            
            Ask questions about machine learning and AI research papers from arXiv.
            The system will search through indexed papers and provide answers with sources.
            """
        )

        with gr.Tabs():
            with gr.Tab("💬 RAG Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What are transformers in machine learning?",
                            lines=2,
                            max_lines=5,
                        )

                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Advanced Options", open=False):
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Number of chunks to retrieve",
                                info="More chunks = more context but slower generation",
                            )

                            use_hybrid = gr.Checkbox(
                                value=True,
                                label="Use hybrid search (BM25 + vector embeddings)",
                                info="Usually better results than keyword-only search",
                            )

                            model_choice = gr.Dropdown(
                                choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                                value=DEFAULT_MODEL,
                                label="LLM Model",
                                info="Larger models may give better answers but are slower",
                            )

                            categories = gr.Textbox(
                                label="arXiv Categories (optional)",
                                placeholder="cs.AI, cs.LG, cs.CL",
                                info="Comma-separated. Leave empty for all categories",
                            )

                response_output = gr.Markdown(
                    label="Answer", value="Ask a question to get started!", height=400, elem_classes=["response-markdown"]
                )

                gr.Examples(
                    examples=[
                        ["What are transformers in machine learning?", 3, True, "llama3.2:1b", "cs.AI, cs.LG"],
                        ["How do convolutional neural networks work?", 5, True, "llama3.2:1b", "cs.CV, cs.LG"],
                        ["What is attention mechanism in deep learning?", 4, False, "llama3.2:1b", "cs.AI"],
                        ["Explain reinforcement learning algorithms", 3, True, "llama3.2:1b", "cs.LG, cs.AI"],
                        ["What are the latest developments in NLP?", 5, True, "llama3.2:1b", "cs.CL"],
                    ],
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                )

                submit_btn.click(
                    fn=stream_response,
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                    outputs=[response_output],
                    show_progress=True,
                )

                query_input.submit(
                    fn=stream_response,
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                    outputs=[response_output],
                    show_progress=True,
                )

            with gr.Tab("🕵️ Skeptic Review"):
                gr.Markdown(
                    """
                    Use this mode to critically review an AI/CS research paper, topic, or claim.
                    The agent retrieves evidence, highlights limitations and unsupported claims, and assigns a risk score.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        skeptic_query = gr.Textbox(
                            label="Paper, claim, or topic to review",
                            placeholder="Review the evidence behind transformer scaling claims",
                            lines=3,
                            max_lines=8,
                        )
                        focus_area = gr.Textbox(
                            label="Focus area (optional)",
                            placeholder="limitations, unsupported claims, baselines, methodology",
                        )

                    with gr.Column(scale=1):
                        skeptic_btn = gr.Button("Run Skeptic Review", variant="primary", size="lg")

                with gr.Accordion("Skeptic Review Options", open=False):
                    skeptic_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of chunks to retrieve",
                        info="Use more chunks for broader evidence coverage",
                    )

                    skeptic_use_hybrid = gr.Checkbox(
                        value=True,
                        label="Use hybrid search (BM25 + vector embeddings)",
                        info="Usually better for finding claim-related evidence",
                    )

                    skeptic_model = gr.Dropdown(
                        choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                        value=DEFAULT_MODEL,
                        label="LLM Model",
                        info="Larger models may provide stronger critiques but are slower",
                    )

                    skeptic_categories = gr.Textbox(
                        label="arXiv Categories (optional)",
                        placeholder="cs.AI, cs.LG, cs.CL",
                        info="Comma-separated. Leave empty for all categories",
                    )

                skeptic_output = gr.Markdown(
                    label="Skeptic Review",
                    value="Enter a paper, claim, or topic to start a skeptical review.",
                    height=600,
                    elem_classes=["response-markdown"],
                )

                gr.Examples(
                    examples=[
                        [
                            "Review the evidence behind transformer scaling claims",
                            "limitations and unsupported claims",
                            5,
                            True,
                            "llama3.2:1b",
                            "cs.AI, cs.LG",
                        ],
                        [
                            "Critique claims that retrieval augmented generation reduces hallucination",
                            "methodology and baselines",
                            5,
                            True,
                            "llama3.2:1b",
                            "cs.CL, cs.AI",
                        ],
                    ],
                    inputs=[skeptic_query, focus_area, skeptic_top_k, skeptic_use_hybrid, skeptic_model, skeptic_categories],
                )

                skeptic_btn.click(
                    fn=skeptic_review_response,
                    inputs=[skeptic_query, focus_area, skeptic_top_k, skeptic_use_hybrid, skeptic_model, skeptic_categories],
                    outputs=[skeptic_output],
                    show_progress=True,
                )

                skeptic_query.submit(
                    fn=skeptic_review_response,
                    inputs=[skeptic_query, focus_area, skeptic_top_k, skeptic_use_hybrid, skeptic_model, skeptic_categories],
                    outputs=[skeptic_output],
                    show_progress=True,
                )

        gr.Markdown(
            """
            ---
            
            **Note**: Make sure the RAG API server is running at `http://localhost:8000` before using this interface.
            
            **Categories**: cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computational Linguistics), 
            cs.CV (Computer Vision), cs.NE (Neural Networks), stat.ML (Statistics - Machine Learning)
            """
        )

    return interface


def main():
    """Main entry point for the Gradio app"""
    print("🚀 Starting arXiv Paper Curator Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed to avoid port conflict
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
