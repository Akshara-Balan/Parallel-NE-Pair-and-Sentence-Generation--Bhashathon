# src/app.py
import streamlit as st
import pandas as pd
from preprocess import load_bpcc_data
from ne_alignment import extract_named_entities, align_entities

st.title("Parallel Named Entity Alignment")

file_path = st.text_input("Enter BPCC dataset path:")
if file_path:
    data = load_bpcc_data(file_path)
    st.dataframe(data.head())

    sentence_idx = st.number_input("Select Sentence Index", min_value=0, max_value=len(data)-1, step=1)
    en_sentence = data.iloc[sentence_idx]["en"]
    target_sentence = data.iloc[sentence_idx]["target"]

    st.write(f"**English:** {en_sentence}")
    st.write(f"**Target:** {target_sentence}")

    en_entities = extract_named_entities(en_sentence, "en")
    target_entities = extract_named_entities(target_sentence, "ml")

    st.write(f"**English NEs:** {en_entities}")
    st.write(f"**Target NEs:** {target_entities}")

    aligned_entities = align_entities(en_sentence, target_sentence)
    st.write(f"**Aligned Entities:** {aligned_entities}")
