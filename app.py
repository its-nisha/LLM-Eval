import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="RQ1 Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("RQ1: LLM Response Outcome & Cross-Lingual Evaluation")
st.caption("Evaluation Model: Gemma-2 | Evaluated Candidate: DeepSeek-R1 (N=582 Valid / 600 Total)")

# Top-level metrics cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Prompts", "600")
c2.metric("Valid Evaluated (N)", "582")
c3.metric("EN Hard Refusals", "14.4% (41)")
c4.metric("ZH Hard Refusals", "0.0% (0)")

st.divider()

# Tab navigation
tab_vis, tab_data, tab_summary = st.tabs(["📊 Visualizations", "📋 Response Explorer", "📄 Summary Report"])

# --- Tab 1: Visualizations ---
with tab_vis:
    st.subheader("1. Outcome Contingency & Language Distribution")
    if os.path.exists("rq1_contingency_plots_20260808_225806.png"):
        st.image("rq1_contingency_plots_20260808_225806.png", use_container_width=True)
    else:
        st.warning("Contingency plot image not found.")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. Refusal & Evasion Rate by Framing (F1–F5)")
        if os.path.exists("rq1_evasion_by_framing_20260808_225806.png"):
            st.image("rq1_evasion_by_framing_20260808_225806.png", use_container_width=True)
        else:
            st.warning("Framing plot image not found.")
            
    with col_b:
        st.subheader("3. Outcome Distribution by Historical Event")
        if os.path.exists("rq1_event_matrix_20260808_225806.png"):
            st.image("rq1_event_matrix_20260808_225806.png", use_container_width=True)
        else:
            st.warning("Event matrix image not found.")

# --- Tab 2: Interactive Data Explorer ---
with tab_data:
    st.subheader("Explore Raw Responses & Judge Rationales")
    csv_file = "rq1_results_20260808_225806.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        # Interactive filter sidebar/columns
        f1, f2, f3 = st.columns(3)
        languages = df["prompt_language"].dropna().unique().tolist()
        sel_lang = f1.multiselect("Language", options=languages, default=languages)
        
        labels = sorted(df["judge_label"].dropna().unique().tolist())
        label_names = {0: "Full Answer (0)", 1: "Hard Refusal (1)", 2: "Soft Evasion (2)"}
        sel_label = f2.multiselect(
            "Judge Label", 
            options=labels, 
            default=labels, 
            format_func=lambda x: label_names.get(x, str(x))
        )
        
        framings = sorted(df["framing"].dropna().unique().tolist())
        sel_framing = f3.multiselect("Framing (F1–F5)", options=framings, default=framings)
        
        filtered = df[
            (df["prompt_language"].isin(sel_lang)) &
            (df["judge_label"].isin(sel_label)) &
            (df["framing"].isin(sel_framing))
        ]
        
        st.write(f"Displaying **{len(filtered)}** of {len(df)} records:")
        st.dataframe(
            filtered[[
                "prompt_id", "prompt_language", "framing", "judge_label", 
                "prompt_text", "cleaned_candidate_response", "judge_reasoning"
            ]],
            use_container_width=True,
            height=480
        )
    else:
        st.error(f"'{csv_file}' not found.")

# --- Tab 3: Text Report ---
with tab_summary:
    st.subheader("Summary Report")
    summary_file = "rq1_descriptive_summary_20260808_225806.txt"
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.warning("Summary text file not found.")
