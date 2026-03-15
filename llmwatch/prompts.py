"""
llmwatch/prompts.py

Agent-style prompt templates for LLMWatch testing.
Each prompt includes a system role and user message
mirroring how real LLM applications work.
"""

SUPPLY_CHAIN_AGENT = {
    "system": "You are an expert supply chain analyst. "
              "Analyze disruptions and recommend actions: "
              "WAIT, EXPEDITE, or SWITCH supplier. "
              "Be concise and decisive.",
    "prompts": [
        "Supplier delay of 5 days detected. Inventory covers 3 days. What do you recommend?",
        "Port congestion reported. Shipment delayed by 2 weeks. Current stock is sufficient for 10 days.",
        "Supplier downtime reported for 48 hours. We have safety stock for 7 days.",
        "Weather disruption affecting carrier routes. Expected delay of 3 days. Stock is critical.",
        "Customs delay of 4 days reported. Production scheduled in 2 days. What action?",
    ]
}

FINANCIAL_AGENT = {
    "system": "You are a financial risk analyst specializing "
              "in supply chain cost impact. Quantify the "
              "financial impact of supply disruptions concisely.",
    "prompts": [
        "A 5-day supplier delay affects 1000 units at $50 each. Calculate impact.",
        "Expediting shipment costs $5000 extra. Production halt costs $2000/day. Evaluate.",
        "Switching suppliers adds $10 per unit on 500 units. Current delay costs $8000. Advise.",
        "Emergency air freight costs $15000. Sea freight delay costs $3000/day for 7 days. Compare.",
    ]
}

RISK_AGENT = {
    "system": "You are a supply chain risk assessment agent. "
              "Evaluate disruption probability and severity. "
              "Score risk from 1-10 and explain briefly.",
    "prompts": [
        "Single supplier for critical component. No backup. Supplier in high-risk region.",
        "Three alternative suppliers available. Primary delayed. Inventory buffer of 14 days.",
        "Weather disruption in supplier region. Historical recovery time 3-5 days.",
        "Supplier financial instability reported. No formal backup plan exists.",
    ]
}


def build_messages(agent: dict, prompt_index: int) -> list:
    """
    Build a full messages list for a given agent and prompt.

    Args:
        agent:        one of the agent dicts above
        prompt_index: which prompt to use

    Returns:
        list of messages ready for LLMWatch.call()
    """
    prompts = agent["prompts"]
    prompt  = prompts[prompt_index % len(prompts)]

    return [
        {"role": "system", "content": agent["system"]},
        {"role": "user",   "content": prompt}
    ]


def rotate_agents() -> list[tuple[str, dict]]:
    """
    Returns all agents as (name, agent) tuples
    for rotating through in tests.
    """
    return [
        ("supply_chain", SUPPLY_CHAIN_AGENT),
        ("financial",    FINANCIAL_AGENT),
        ("risk",         RISK_AGENT),
    ]