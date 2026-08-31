from __future__ import annotations

import os
import uuid

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("COPILOT_DEVICE", "cpu")

import streamlit as st

from copilot.agent import Agent
from copilot.retrieval.query import build_query

st.set_page_config(page_title="Shopping assistant")


@st.cache_resource(show_spinner="Loading models + catalog...")
def get_agent() -> Agent:
    return Agent("data/catalog.jsonl")


agent = get_agent()

with st.sidebar:
    st.header("Session")
    if st.button("New conversation", use_container_width=True):
        for k in ("sid", "turn", "history"):
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.caption("Profile (applied to a new conversation)")
    rating_style = st.selectbox("rating_style", ["usually positive", "critical"])
    freq = st.selectbox(
        "purchase_frequency",
        ["3-4 prior purchases", "1 prior purchase", "first-time buyer"],
    )
    tags_raw = st.text_input("preference_tags", "fit, comfort, durability")

if "sid" not in st.session_state:
    st.session_state.sid = f"ui_{uuid.uuid4().hex}"
    st.session_state.turn = 0
    st.session_state.history = []  # list[(role, message, recs)]
    agent.reset(
        st.session_state.sid,
        {
            "rating_style": rating_style,
            "average_prior_rating": 3.0 if rating_style == "critical" else 4.5,
            "purchase_frequency": freq,
            "preference_tags": [t.strip() for t in tags_raw.split(",") if t.strip()],
            "summary": "",
        },
    )

st.title("Shopping assistant")


def _considering(sid: str, turn: int, resp: dict) -> list[tuple[str, str]]:
    state = agent._sessions.get(sid)
    if state is None:
        return []
    q = build_query(state, None)
    slots = "  ".join(
        f"{a}={s.value!r}({s.confidence:.2f},{s.source})" for a, s in state.slots.items()
    ) or "-"
    prof = state.user_profile
    rows = [
        ("turn", str(turn)),
        ("track / p_buying", f"{q.track}  {state.intent_p_buying:.2f}"),
        ("free_text (-> BM25 + CE)", q.free_text or "-"),
        ("slots", slots),
        ("soft slots", " ".join(f"{a}={s.value}" for a, s in q.slots.items()
                                if a in ("color", "material", "brand")) or "-"),
        ("negations", str(dict(q.negations)) if q.negations else "-"),
        ("disclosed phrases", str(list(state.disclosed_phrases)) or "-"),
        ("ask this turn", str(resp.get("ask_attribute"))),
        ("asked so far", str(sorted(state.asked_attributes)) or "-"),
        ("no-preference / clarify#", f"{sorted(state.no_preference)}  n={state.clarify_count}"),
        ("pending relaxation", str(state.pending_relaxation) if state.pending_relaxation else "-"),
        ("shown asins", str(len(state.shown_asins))),
        ("profile (-> profile_calib)",
         f"rating_style={getattr(prof, 'rating_style', None)!r} "
         f"tags={getattr(prof, 'preference_tags', None)} "
         f"freq={getattr(prof, 'purchase_frequency', None)!r}"),
    ]
    top3 = []
    for rec in resp.get("recommendations", [])[:3]:
        c = agent.catalog.get(rec["parent_asin"])
        top3.append(f"{rec['parent_asin']} {c.title[:50] if c else '?'}")
    rows.append(("top 3", "  |  ".join(top3) or "-"))

    print(f"\n{'-'*4} agent considering (session {sid[:12]}, turn {turn}) {'-'*4}", flush=True)
    for label, value in rows:
        print(f"  {label:26s}: {value}", flush=True)
    return rows


def _render_recs(recs: list[dict]) -> None:
    for i, rec in enumerate(recs, 1):
        c = agent.catalog.get(rec["parent_asin"])
        if c is None:
            continue
        price = f"${c.price:.2f}" if c.price else "-"
        cats = " / ".join(c.categories[-2:]) if c.categories else ""
        st.markdown(
            f"**{i}. {c.title[:100]}**  \n"
            f"{c.brand or '-'} - {price} - {c.average_rating:.1f} stars "
            f"({c.rating_number}) - {cats}  \n"
            f"`{rec['parent_asin']}`"
        )


for role, message, recs in st.session_state.history:
    with st.chat_message(role):
        st.write(message)
        if recs:
            _render_recs(recs)

if user_msg := st.chat_input("What are you looking for?"):
    st.session_state.turn += 1
    with st.chat_message("user"):
        st.write(user_msg)
    st.session_state.history.append(("user", user_msg, None))

    resp = agent.respond(st.session_state.sid, user_msg, st.session_state.turn, 10)
    reply = resp.get("message", "")
    if resp.get("ask_attribute"):
        reply += f"  \n\n_asking about: **{resp['ask_attribute']}**_"
    recs = resp.get("recommendations", [])
    rows = _considering(st.session_state.sid, st.session_state.turn, resp)  # also -> console

    with st.chat_message("assistant"):
        st.write(reply)
        _render_recs(recs)
        with st.expander("What the agent is considering"):
            st.table({"": [f"**{k}**" for k, _ in rows], " ": [v for _, v in rows]})
    st.session_state.history.append(("assistant", reply, recs))
