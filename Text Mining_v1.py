import re
import json
import string
import streamlit as st
import pandas as pd
import nltk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from community import community_louvain
from wordcloud import WordCloud
from scipy.spatial import ConvexHull
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import Ridge
from textblob import TextBlob
from PIL import Image, ImageDraw
from collections import Counter

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Fragrance Verbatim Lab Pro",
    layout="wide",
    page_icon="🧪"
)

# =============================================================================
# ▌BLOCK 1 — V3 NLP PIPELINE CONSTANTS
# =============================================================================

@st.cache_resource
def setup_nltk():
    nltk.download("wordnet",                        quiet=True)
    nltk.download("omw-1.4",                        quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

CONTRACTION_MAP: dict[str, str] = {
    "don't":   "not",  "doesn't":  "not",  "didn't":   "not",
    "isn't":   "not",  "aren't":   "not",  "wasn't":   "not",
    "weren't": "not",  "can't":    "not",  "couldn't": "not",
    "won't":   "not",  "wouldn't": "not",  "shouldn't":"not",
    "ain't":   "not",  "shan't":   "not",  "haven't":  "not",
    "hasn't":  "not",  "hadn't":   "not",  "mustn't":  "not",
    "needn't": "not",
    "it's":    "it",   "that's":   "that", "he's":     "he",
    "she's":   "she",  "there's":  "there","here's":   "here",
    "i'm":     "i",    "i've":     "i",    "i'll":     "i",
    "i'd":     "i",    "we're":    "we",   "they're":  "they",
    "you're":  "you",  "let's":    "let",  "you'd":    "you",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP) + r")\b",
    flags=re.IGNORECASE,
)

CLAUSE_SPLITTERS = {"but", "however", "although", "though", "yet", "whereas"}

NEGATIVE_PREFIXES: Tuple[str, ...] = ("un", "in", "im", "ir", "il", "non", "dis")

PREFIX_STRIP_PROTECT = {
    "invigorate", "invigorating", "invigoration",
    "inspire", "inspiring", "inspiration", "inspired",
    "intense", "intensity", "intensify",
    "indulge", "indulging", "indulgent", "indulgence",
    "intoxicate", "intoxicating", "intoxication",
    "intimate", "intimacy",
    "incredible", "incredibly",
    "interesting",
    "irritate", "irritating", "irritation", "irritated",
    "irresistible", "irresistibly",
    "illuminate", "illuminating",
    "imagine", "imagining", "imagination",
    "immerse", "immersing", "immersion",
    "impact",
    "distinctive", "distinguish", "display",
}

NEGATION_TERMS = {"not", "never", "no", "without", "nothing", "none", "nobody"}

INTENSITY_MAP: dict[str, str] = {
    "very": "very", "so": "very", "really": "very", "extremely": "very",
    "incredibly": "very", "super": "very", "highly": "very", "deeply": "very",
    "absolutely": "very", "totally": "very", "quite": "very", "pretty": "very",
    "awfully": "very", "terribly": "very", "remarkably": "very",
}

NEUTRAL_STOPS = {
    "smell", "smells", "smelling", "scent", "scents", "odor", "odour",
    "perfume", "fragrance", "aroma", "note", "nuance", "touch", "hint", "product",
    "feel", "feels", "feeling", "feelings", "felt",
    "impression", "image", "association", "associates", "reminds",
    "remind", "remember", "evoke", "seem", "find",
    "think", "make", "let", "go", "put", "get", "give", "come", "take",
    "look", "say", "use", "suit", "mind",
    "also", "though", "simply", "somehow", "therefore", "order", "almost", "just",
    "little", "lot", "bit", "quite", "really", "more", "less", "enough",
    "most", "few", "many", "some", "any", "all", "both", "either",
    "neither", "one", "two", "each", "same", "other", "own",
    "like", "kind", "kinda", "real", "something", "anything", "everything", "nothing",
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "could", "should", "may", "might", "shall", "can", "cannot", "ought",
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with", "by", "as",
    "from", "into", "about", "above", "after", "again", "against",
    "before", "below", "between", "down", "during", "off", "once",
    "only", "out", "over", "through", "under", "up", "while",
    "and", "or", "if", "that", "this", "these", "those",
    "there", "here", "than", "then", "such", "so",
    "who", "whom", "which", "when", "where", "why", "how", "what",
    "because", "until", "further", "too", "very",
    "therefore", "hence", "thus", "yet", "still", "already",
    "indeed", "perhaps", "maybe",
}

FRAGRANCE_MERGES = {
    "flowery": "flower", "flowers": "flower", "blooming": "flower",
    "blossom": "flower", "blossomy": "flower",
    "freshness": "fresh", "freshly": "fresh",
    "cleanliness": "clean", "cleaning": "clean",
    "relaxed": "relax", "relaxing": "relax", "relaxation": "relax",
    "comforting": "comfort", "comforted": "comfort", "comfortable": "comfort",
    "woodsy": "woody",
    "fruity": "fruit",
    "spicy": "spice", "spiced": "spice",
    "musky": "musk",
    "sweetness": "sweet",
    "marine": "ocean", "sea": "ocean", "oceanic": "ocean",
    "watery": "aquatic", "ozonic": "aquatic",
    "powdery": "powder", "powdered": "powder",
    "smoky": "smoke", "smokey": "smoke",
    "grassy": "green", "herbal": "green", "herbaceous": "green", "leafy": "green",
    "grandma": "grandmother", "grandpa": "grandfather",
    "old fashion": "old_fashioned",
}

EMOTION_BIGRAMS = {("feel", "good"): "feel_good"}

WEIGHT_VERY  = 1.5
WEIGHT_PLAIN = 1.0

_PUNCT_STRIP = re.compile(r"[" + re.escape(string.punctuation.replace("-", "")) + r"]")


# =============================================================================
# ▌BLOCK 2 — V3 PIPELINE STEPS
# =============================================================================

def step1_char_normalize(text: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

def step2_expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(lambda m: CONTRACTION_MAP[m.group(0).lower()], text)

def step3_clause_segment(text: str) -> List[str]:
    pattern = r"(?:,\s*|\s+)(?:" + "|".join(CLAUSE_SPLITTERS) + r")(?:\s+|,\s*)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

def step4_tokenize(text: str) -> List[str]:
    return [t for t in _PUNCT_STRIP.sub("", text).split() if t]

def step5_prefix_strip(tokens: List[str]) -> List[str]:
    result = []
    for tok in tokens:
        if tok in PREFIX_STRIP_PROTECT:
            result.append(tok); continue
        stripped = False
        for pref in NEGATIVE_PREFIXES:
            if tok.startswith(pref) and len(tok) > len(pref) + 2:
                result.extend(["not", tok[len(pref):]]);stripped = True; break
        if not stripped: result.append(tok)
    return result

def _get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "V": wordnet.VERB,
            "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def step6_lemmatize(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if tok in NEGATION_TERMS or tok in INTENSITY_MAP:
            out.append(tok)
        else:
            out.append(lemmatizer.lemmatize(tok, pos=_get_wordnet_pos(tok)))
    return out

def step5_5_synonym_merge(tokens: List[str]) -> List[str]:
    return [FRAGRANCE_MERGES.get(tok, tok) for tok in tokens]

def step7_stopword_removal(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t in NEGATION_TERMS or t in INTENSITY_MAP
            or "_" in t or t not in NEUTRAL_STOPS]

def step7_5_intensity_normalize(tokens: List[str]) -> List[str]:
    return [INTENSITY_MAP.get(t, t) for t in tokens]

def step_feel_good_bigram(tokens: List[str]) -> List[str]:
    out = []; i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            pair = (tokens[i], tokens[i+1])
            if pair in EMOTION_BIGRAMS:
                out.append(EMOTION_BIGRAMS[pair]); i += 2; continue
        out.append(tokens[i]); i += 1
    return out

def step8_ngram_collapse(tokens: List[str]) -> List[str]:
    out = []; i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "_" in tok:
            out.append(tok); i += 1; continue
        if tok == "not" and i + 1 < len(tokens):
            nxt = tokens[i+1]
            if nxt == "very" and i + 2 < len(tokens):
                tgt = tokens[i+2]
                if "_" not in tgt and tgt not in NEGATION_TERMS:
                    out.append(f"not_{tgt}"); i += 3; continue
            if nxt not in NEGATION_TERMS and "_" not in nxt:
                out.append(f"not_{nxt}"); i += 2; continue
        if tok == "very" and i + 1 < len(tokens):
            nxt = tokens[i+1]
            if nxt not in NEGATION_TERMS and nxt != "very" and "_" not in nxt:
                out.append(f"very_{nxt}"); i += 2; continue
        out.append(tok); i += 1
    return out

def step9_dedup(tokens: List[str]) -> List[str]:
    seen = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]

def process_verbatim(raw_text: str) -> List[str]:
    if not raw_text or not isinstance(raw_text, str): return []
    text = step1_char_normalize(raw_text)
    text = step2_expand_contractions(text)
    all_tokens = []
    for clause in step3_clause_segment(text):
        tokens = step4_tokenize(clause)
        tokens = step5_prefix_strip(tokens)
        tokens = step6_lemmatize(tokens)
        tokens = step5_5_synonym_merge(tokens)
        tokens = step_feel_good_bigram(tokens)
        tokens = step7_stopword_removal(tokens)
        tokens = step7_5_intensity_normalize(tokens)
        tokens = step8_ngram_collapse(tokens)
        all_tokens.extend(tokens)
    return step9_dedup(all_tokens)

def tokens_to_string(tokens: List[str]) -> str:
    """Convert token list to space-joined string for vectorizers.
    Underscores are preserved so very_pleasant / not_clean stay as single tokens.
    """
    return " ".join(tokens)

def get_weight(token: str) -> float:
    return WEIGHT_VERY if token.startswith("very_") else WEIGHT_PLAIN

def get_bucket(token: str) -> str:
    if token.startswith("not_"):  return "negative"
    if token.startswith("very_"): return "positive_amplified"
    return "positive"


# =============================================================================
# ▌BLOCK 3 — VISUALISATION HELPERS (from V1, adapted)
# =============================================================================

def generate_word_cloud(token_series: pd.Series, palette: str, shape: str):
    """Build weighted word cloud from token lists.
    Tokens like very_pleasant / not_clean are kept as single units for frequency,
    but displayed with spaces for readability.
    """
    weight_counter: Counter = Counter()
    for tokens in token_series:
        if isinstance(tokens, list):
            for t in tokens:
                display = t.replace("_", " ")
                weight_counter[display] += get_weight(t)
    if not weight_counter:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No text available", ha="center"); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20, 20, 780, 780), fill=0)
        mask = np.array(img)
    wc = WordCloud(
        background_color="white", colormap=palette, mask=mask,
        width=800, height=500, collocations=False,
        regexp=r"\S+"
    ).generate_from_frequencies(weight_counter)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    return fig


def generate_word_tree_advanced(token_series: pd.Series, min_freq: int, palette: str):
    """Network word tree from token lists."""
    texts = [tokens_to_string(t) for t in token_series if isinstance(t, list) and t]
    if not texts: return None
    try:
        vec = CountVectorizer(min_df=min_freq, token_pattern=r"(?u)\b\S+\b")
        mtx = vec.fit_transform(texts)
        words = vec.get_feature_names_out()
        word_counts = np.asarray(mtx.sum(axis=0)).flatten()
        count_dict = dict(zip(words, word_counts))
        if len(words) < 2: return None
        adj = mtx.T * mtx; adj.setdiag(0)
        G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        G.remove_nodes_from(list(nx.isolates(G)))
        if len(G.nodes) < 2: return None
        partition = community_louvain.best_partition(G)
        pos = nx.spring_layout(G, k=0.3, seed=42, iterations=500)
        fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
        ax.set_facecolor("white")
        PASTEL = ["#A8D8B9","#F4B8C1","#B5D0E8","#D4E8A8","#C8B8E8","#F4D8A8","#A8D8D8","#E8C8B8"]
        for i, comm in enumerate(sorted(set(partition.values()))):
            nodes = [n for n in G.nodes() if partition[n] == comm]
            if not nodes: continue
            pts = np.array([pos[n] for n in nodes])
            color = PASTEL[i % len(PASTEL)]
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    ax.add_patch(patches.Polygon(
                        pts[hull.vertices], closed=True,
                        facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5, zorder=0))
                except: pass
            else:
                ax.add_artist(plt.Circle(np.mean(pts, axis=0), 0.1, color=color, alpha=0.2, zorder=0))
        nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="#aaaaaa", ax=ax)
        max_c = max(word_counts)
        for node, (x, y) in pos.items():
            fsize = 10 + (count_dict[node] / max_c) * 20
            ax.text(x, y, node, fontsize=fsize, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.3),
                    color="#222222", zorder=3)
        plt.axis("off")
        return fig
    except: return None


def run_fca(df: pd.DataFrame, p_col: str, fmin: int, use_tfidf: bool):
    grouped = df.groupby(p_col)["token_str"].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products for Factorial Mapping."
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=min(fmin, len(grouped)), token_pattern=r"(?u)\b\S+\b")
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    X_centered = X - np.mean(X, axis=0)
    svd = TruncatedSVD(n_components=2, random_state=42)
    row_coords = svd.fit_transform(X_centered)
    col_coords = svd.components_.T * (np.std(row_coords) / (np.std(svd.components_.T) + 1e-9))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_), None


# =============================================================================
# ▌BLOCK 4 — SIDEBAR & FILE LOADING
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

    fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
    use_tfidf   = st.toggle("Use TF-IDF Weighting", value=True)
    shape_opt   = st.radio("Cloud Shape", ["Rectangle", "Round"])
    palette_opt = st.selectbox("Palette", ["copper","GnBu","RdPu","viridis","Spectral"])

    if uploaded_file:
        try:
            xl     = pd.ExcelFile(uploaded_file)
            sheet  = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)

            filter_col  = st.selectbox("Filter Column:", ["No Filter"] + list(df_raw.columns))
            target_indices = df_raw.index
            filter_label   = "Total Sample"

            if filter_col != "No Filter":
                options = sorted(df_raw[filter_col].dropna().unique())
                selected_codes = st.multiselect("Select Codes:", options)
                if selected_codes:
                    target_indices = df_raw[df_raw[filter_col].isin(selected_codes)].index
                    filter_label   = f"{filter_col}: {', '.join(map(str, selected_codes))}"

            p_col = st.selectbox("Product ID Column",  df_raw.columns)
            v_col = st.selectbox("Verbatim Column",    df_raw.columns)
            s_col = st.selectbox("Preference Score (Optional)", ["None"] + list(df_raw.columns))

        except Exception as e:
            st.error(f"Error loading file: {e}"); st.stop()

    if uploaded_file and st.button("🚀 Run Analysis"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col]).copy()

        with st.spinner("Running V3 pipeline…"):
            df_filtered["tokens"]    = df_filtered[v_col].apply(process_verbatim)
            df_filtered["token_str"] = df_filtered["tokens"].apply(tokens_to_string)

        st.session_state["processed_df"] = df_filtered
        st.session_state["filter_info"]  = filter_label
        st.session_state["pref_col"]     = s_col
        st.session_state["p_col"]        = p_col
        st.session_state["v_col"]        = v_col
        st.success(f"✅ Processed {len(df_filtered)} verbatims")


# =============================================================================
# ▌BLOCK 5 — TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs([
    "📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map",
    "🔍 Topic Lab", "🎯 Impact Lab", "🚫 Exclusions & Grams"
])

if "processed_df" in st.session_state:
    df    = st.session_state["processed_df"]
    p_col = st.session_state["p_col"]
    v_col = st.session_state["v_col"]
    p_list = sorted(df[p_col].dropna().astype(str).unique())

    # ── Tab 1: Single Product ─────────────────────────────────────────────
    with tab1:
        target_p     = st.selectbox("Fragrance Focus", p_list)
        product_data = df[df[p_col].astype(str) == target_p]

        sent_val = product_data[v_col].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity).mean()
        st.metric(
            f"Mood: {target_p}",
            "Positive" if sent_val > 0 else "Negative",
            f"{round(sent_val * 100, 1)}%"
        )

        st.write("### 🌳 Olfactive Word Tree")
        tree_fig = generate_word_tree_advanced(
            product_data["tokens"], fmin_global, palette_opt)
        if tree_fig: st.pyplot(tree_fig)
        else: st.warning("Not enough data for tree with current Min Frequency setting.")

        st.divider()
        st.write("### ☁️ Weighted Wordcloud")
        st.pyplot(generate_word_cloud(product_data["tokens"], palette_opt, shape_opt))

        st.divider()
        st.write("### 🪣 Token Buckets")
        bucket_rows = []
        for tokens in product_data["tokens"]:
            if isinstance(tokens, list):
                for t in tokens:
                    bucket_rows.append({"token": t, "bucket": get_bucket(t), "weight": get_weight(t)})
        if bucket_rows:
            bdf = pd.DataFrame(bucket_rows)
            col_pos, col_amp, col_neg = st.columns(3)
            col_pos.write("**Positive**")
            col_pos.dataframe(
                bdf[bdf["bucket"]=="positive"].groupby("token")["weight"]
                .sum().sort_values(ascending=False).reset_index(), use_container_width=True)
            col_amp.write("**Positive Amplified (very_)**")
            col_amp.dataframe(
                bdf[bdf["bucket"]=="positive_amplified"].groupby("token")["weight"]
                .sum().sort_values(ascending=False).reset_index(), use_container_width=True)
            col_neg.write("**Negative (not_)**")
            col_neg.dataframe(
                bdf[bdf["bucket"]=="negative"].groupby("token")["weight"]
                .sum().sort_values(ascending=False).reset_index(), use_container_width=True)

    # ── Tab 2: Comparison ─────────────────────────────────────────────────
    with tab2:
        st.subheader("⚔️ Scent Comparison")
        comp_cols = st.columns(2)
        p_a = comp_cols[0].selectbox("Fragrance A", p_list, index=0)
        p_b = comp_cols[1].selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
        d_a = df[df[p_col].astype(str) == p_a]["token_str"]
        d_b = df[df[p_col].astype(str) == p_b]["token_str"]
        if not d_a.empty and not d_b.empty:
            sim = float(cosine_similarity(
                TfidfVectorizer(token_pattern=r"(?u)\b\S+\b")
                .fit_transform([" ".join(d_a), " ".join(d_b)]))[0][1])
            st.metric("Olfactive Similarity", f"{round(sim * 100, 1)}%")
            tok_a = df[df[p_col].astype(str) == p_a]["tokens"]
            tok_b = df[df[p_col].astype(str) == p_b]["tokens"]
            comp_cols[0].pyplot(generate_word_cloud(tok_a, palette_opt, shape_opt))
            comp_cols[1].pyplot(generate_word_cloud(tok_b, palette_opt, shape_opt))

    # ── Tab 3: Factorial Map ──────────────────────────────────────────────
    with tab3:
        st.subheader("🌐 Factorial Mapping")
        res, err = run_fca(df, p_col, fmin_global, use_tfidf)
        if not err:
            r_c, c_c, prods, wrds, _ = res
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(r_c[:,0], r_c[:,1], c="blue", s=100)
            for i, txt in enumerate(prods):
                ax.text(r_c[i,0], r_c[i,1], txt, fontsize=12)
            ax.scatter(c_c[:,0], c_c[:,1], c="red", marker="x", alpha=0.2)
            norms = [np.linalg.norm(c) for c in c_c]
            threshold = np.percentile(norms, 80)
            for i, txt in enumerate(wrds):
                if norms[i] > threshold:
                    ax.text(c_c[i,0], c_c[i,1], txt, color="darkred", fontsize=8)
            st.pyplot(fig)
        else:
            st.error(err)

    # ── Tab 4: Topic Lab ──────────────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Topic Lab")
        num_t = st.slider("Themes", 2, 8, 3)
        if st.button("Generate Topics"):
            vec  = TfidfVectorizer(max_features=500, token_pattern=r"(?u)\b\S+\b")
            mtx  = vec.fit_transform(df["token_str"])
            nmf  = NMF(n_components=num_t, random_state=42, init="nndsvd").fit(mtx)
            fn   = vec.get_feature_names_out()
            cols = st.columns(num_t)
            for i, topic in enumerate(nmf.components_):
                top_words = [fn[j] for j in topic.argsort()[-7:]]
                with cols[i % num_t]:
                    st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))

    # ── Tab 5 (tab6): Impact Lab ──────────────────────────────────────────
    with tab6:
        st.subheader("🎯 Preference Driver Analysis")
        pref_col = st.session_state.get("pref_col", "None")
        if pref_col != "None":
            try:
                df_imp = df.dropna(subset=[pref_col, "token_str"])
                df_imp = df_imp[df_imp["token_str"] != ""]
                vec_imp = CountVectorizer(min_df=3, binary=True, token_pattern=r"(?u)\b\S+\b")
                X_imp   = vec_imp.fit_transform(df_imp["token_str"])
                y_imp   = df_imp[pref_col]
                model   = Ridge(alpha=1.0).fit(X_imp, y_imp)
                impact_df = pd.DataFrame({
                    "Word":   vec_imp.get_feature_names_out(),
                    "Impact": model.coef_
                }).sort_values("Impact", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("📈 Positive Drivers")
                    top10 = impact_df.head(10)
                    fig_pos, ax_pos = plt.subplots(figsize=(5, 4))
                    ax_pos.barh(top10["Word"], top10["Impact"], color="steelblue")
                    ax_pos.invert_yaxis(); plt.tight_layout(); st.pyplot(fig_pos)
                with c2:
                    st.write("📉 Negative Drivers")
                    bot10 = impact_df.tail(10)
                    fig_neg, ax_neg = plt.subplots(figsize=(5, 4))
                    ax_neg.barh(bot10["Word"], bot10["Impact"], color="salmon")
                    ax_neg.invert_yaxis(); plt.tight_layout(); st.pyplot(fig_neg)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Select a Preference Score column in the sidebar to enable this tab.")

    # ── Tab 6 (tab5): Exclusions & Grams ─────────────────────────────────
    with tab5:
        st.subheader("🚫 Exclusions & Synonym Lab")
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("**Current Stopwords**")
            stops_text = st.text_area(
                "Edit stopwords (comma-separated)",
                value=", ".join(sorted(NEUTRAL_STOPS)),
                height=200
            )

        with col_right:
            st.write("**Fragrance Merges**")
            merges_text = st.text_area(
                "Edit merges (one per line: variant → canonical)",
                value="\n".join(f"{k} → {v}" for k, v in sorted(FRAGRANCE_MERGES.items())),
                height=200
            )

        st.write("**Prefix Strip Protect**")
        protect_text = st.text_area(
            "Edit protected words (comma-separated)",
            value=", ".join(sorted(PREFIX_STRIP_PROTECT)),
            height=100
        )

        if st.button("💾 Apply & Re-Process"):
            # Update stopwords
            new_stops = {x.strip().lower() for x in stops_text.split(",") if x.strip()}
            NEUTRAL_STOPS.clear(); NEUTRAL_STOPS.update(new_stops)
            # Update merges
            new_merges = {}
            for line in merges_text.splitlines():
                if "→" in line:
                    parts = line.split("→", 1)
                    if len(parts) == 2:
                        new_merges[parts[0].strip().lower()] = parts[1].strip().lower()
            FRAGRANCE_MERGES.clear(); FRAGRANCE_MERGES.update(new_merges)
            # Update protect
            new_protect = {x.strip().lower() for x in protect_text.split(",") if x.strip()}
            PREFIX_STRIP_PROTECT.clear(); PREFIX_STRIP_PROTECT.update(new_protect)
            st.success("Settings updated — click 🚀 Run Analysis to re-process.")

else:
    for tab in [tab1, tab2, tab3, tab4, tab6, tab5]:
        with tab:
            st.info("⬅️ Upload an Excel file and click **🚀 Run Analysis** to get started.")