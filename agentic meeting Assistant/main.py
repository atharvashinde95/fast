"""
main.py — Entry Point for the Meeting Intelligence System

Usage:
    python main.py                          # uses built-in sample transcript
    python main.py --file path/to/file.txt  # provide your own transcript file

Flow:
    1. Load transcript
    2. Run extraction graph (one shot)
    3. Display structured results
    4. Enter Q&A loop (type questions, 'quit' to exit)
"""

import argparse
import json
import sys
from langchain_core.messages import HumanMessage
from graph import extraction_graph, qa_graph
from state import initial_state


# ── Sample transcript (used when no file is provided) ─────────────────────────

SAMPLE_TRANSCRIPT = """
Meeting: Q3 Product Roadmap Planning
Date: October 14, 2024
Attendees: Sarah Chen (Product Manager), James Okafor (Engineering Lead), 
           Priya Nair (Designer), Mark Thompson (CEO), Lisa Wang (QA Lead)

Sarah: Alright everyone, let's get started. The main goal today is to finalize 
the Q3 roadmap and make sure we're aligned before we go public with it next week.

Mark: Before we dive in, I want to flag that the board is really pushing for the 
mobile app launch this quarter. That's non-negotiable at this point.

James: Understood. So the mobile app needs to be in the app stores by end of Q3, 
which gives us about 8 weeks. That's tight but doable if we cut scope on the 
analytics dashboard.

Sarah: Agreed. Let's officially descope the advanced analytics features to Q4. 
James, can you update the JIRA board to reflect that by tomorrow?

James: Will do. I'll also need Priya to finalize the mobile UI designs by this 
Friday so the dev team can start implementation on Monday.

Priya: Friday works for me. I'll send the Figma links to James by EOD Friday. 
I do want to flag one thing though — the current color system doesn't meet 
WCAG accessibility standards on mobile. We need to fix that.

Mark: That's a priority. We cannot launch something that fails accessibility. 
Priya, make that your top concern for the mobile designs.

Priya: Got it, accessibility first.

Sarah: Good. Lisa, what does QA need from us to be ready for the mobile launch?

Lisa: We need a staging environment set up at least 3 weeks before launch, so 
by September 23rd. And we need a test plan document from James's team 
covering all the critical user flows.

James: I can have the test plan ready by September 15th. And I'll loop in DevOps 
today to get the staging environment timeline confirmed.

Mark: Perfect. One more thing — we've been getting a lot of requests for Slack 
integration from enterprise customers. I'd like someone to investigate feasibility 
for a Q4 roadmap item.

Sarah: I'll own that investigation. I'll have a feasibility report ready by 
October 30th.

Mark: Great. Sounds like we have a plan. Let's do a quick check-in next Thursday 
to make sure we're on track. Sarah, can you set that up?

Sarah: Sure, I'll send the calendar invite today.

Mark: Perfect. Everyone aligned? Good. Thanks for the productive session.
"""


# ── Display helpers ───────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}")


def print_section(title: str, content):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")

    if isinstance(content, list):
        if not content:
            print("  (none found)")
        for i, item in enumerate(content, 1):
            if isinstance(item, dict):
                for k, v in item.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, list):
                        print(f"  {label}: {', '.join(v)}")
                    else:
                        print(f"  {label}: {v}")
                if i < len(content):
                    print(f"  {'·' * 30}")
            else:
                print(f"  • {item}")
    elif isinstance(content, str):
        # Word-wrap at ~70 chars
        words = content.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 72:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)


def display_extraction_results(state: dict):
    print_header("MEETING INTELLIGENCE REPORT")

    print_section("SUMMARY", state.get("summary", ""))
    print_section("PARTICIPANTS", state.get("participants", []))
    print_section("KEY TOPICS DISCUSSED", state.get("key_topics", []))
    print_section("DECISIONS MADE", state.get("decisions", []))
    print_section("ACTION ITEMS", state.get("action_items", []))


# ── Main ──────────────────────────────────────────────────────────────────────

def load_transcript(file_path: str | None) -> str:
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            print(f"[✓] Loaded transcript from: {file_path}")
            return transcript
        except FileNotFoundError:
            print(f"[✗] File not found: {file_path}")
            sys.exit(1)
    else:
        print("[i] No file provided — using built-in sample transcript.")
        return SAMPLE_TRANSCRIPT


def run_qa_loop(state: dict):
    """Interactive Q&A loop. Maintains full state across turns for memory."""
    print_header("Q&A — Ask Anything About This Meeting")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Session ended]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n[Goodbye! Session ended.]")
            break

        # Append user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the QA graph — it reads state, appends AI response to messages
        state = qa_graph.invoke(state)

        # Print the latest AI response (last message)
        last_msg = state["messages"][-1]
        print(f"\nAssistant: {last_msg.content}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Meeting Intelligence System — Powered by LangGraph"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file containing the meeting transcript",
    )
    args = parser.parse_args()

    print_header("MEETING INTELLIGENCE SYSTEM")
    print("  Powered by LangGraph + Claude\n")

    # Step 1: Load transcript
    transcript = load_transcript(args.file)

    # Step 2: Build initial state
    state = initial_state(transcript)

    # Step 3: Run extraction graph
    print("\n[Graph] Running extraction pipeline...")
    state = extraction_graph.invoke(state)

    if state.get("error"):
        print(f"\n[✗] Extraction failed: {state['error']}")
        sys.exit(1)

    # Step 4: Display results
    display_extraction_results(state)

    # Step 5: Enter Q&A loop
    run_qa_loop(state)


if __name__ == "__main__":
    main()
