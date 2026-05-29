import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import re
import urllib.request
import os
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ── NLTK download ───────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)

# ── Streamlit config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Sentimen · Kelompok 8",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,.12) 0%, rgba(139,92,246,.08) 100%);
        border: 1px solid rgba(139,92,246,.25);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,.08);
        transition: transform .2s, box-shadow .2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,.18);
    }
    div[data-testid="stMetric"] label {
        font-size: .82rem !important;
        letter-spacing: .04em;
        text-transform: uppercase;
        opacity: .75;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 2rem 0 1rem;
        padding-bottom: .5rem;
        border-bottom: 3px solid rgba(99,102,241,.5);
        display: inline-block;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,.3);
    }
    .hero h1 { font-size: 2.2rem; font-weight: 900; margin: 0; }
    .hero p  { font-size: 1.05rem; opacity: .9; margin-top: .5rem; }

    /* ── Plotly chart containers ── */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 16px rgba(0,0,0,.06);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* hide default header */
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

NORMALIZED_WORD_DICT = {
    "bgt": "banget", "yg": "yang", "gk": "tidak", "ga": "tidak",
    "gak": "tidak", "nggak": "tidak", "tdk": "tidak", "kalo": "kalau",
    "kl": "kalau", "pake": "pakai", "pk": "pakai", "smpe": "sampai",
    "smpai": "sampai", "jd": "jadi", "jdi": "jadi", "utk": "untuk",
    "dr": "dari", "krn": "karena", "karna": "karena", "trs": "terus",
    "trus": "terus", "dgn": "dengan", "blm": "belum", "udh": "sudah",
    "sdh": "sudah", "aja": "saja", "doang": "saja", "aj": "saja",
    "org": "orang", "tmn": "teman", "temen": "teman", "bnr": "benar",
    "mantul": "mantap betul", "btw": "ngomong-ngomong", "lol": "lucu",
    "wkwk": "tertawa", "wkwwk": "tertawa", "hehe": "tertawa",
    "xixi": "tertawa", "cm": "cuma", "cmn": "cuma", "tp": "tetapi",
    "tpi": "tetapi", "dlm": "dalam", "jg": "juga", "sy": "saya",
    "gw": "saya", "gue": "saya", "akuh": "aku", "aq": "aku",
    "km": "kamu", "lu": "kamu", "loe": "kamu", "bgtu": "begitu",
    "gitu": "begitu", "gni": "begini", "gin": "begini", "gini": "begini",
    "asbun": "asal bunyi", "ngasbun": "mengasal bunyi", "mkn": "makan",
    "minum2": "minum-minum", "pdhl": "padahal", "dpt": "dapat",
    "bsa": "bisa", "bs": "bisa", "cape": "capek", "capee": "capek",
    "bgus": "bagus", "jele": "jelek", "makasih": "terima kasih",
    "thx": "terima kasih", "makasi": "terima kasih", "sippp": "sip",
    "okk": "oke", "okey": "oke", "udhhlah": "sudahlah",
}

ADDITIONAL_STOPWORDS = {
    "utk", "yg", "nya", "aja", "ga", "gk", "kalo", "gak",
    "nih", "sih", "dong", "deh", "lah", "pun", "saad", "bgt",
    "dan", "isi", "kayak", "ya", "si", "tp", "moga", "dg", "dr",
    "klo", "pake", "rb", "dar", "sy", "lg", "bs", "jg", "krn", "tdk",
    "yuk", "hm", "eh", "ah", "oh", "wah", "oke", "gas",
}

SENTIMENT_COLORS = {
    "negatif": "#ef4444",
    "netral":  "#94a3b8",
    "positif": "#22c55e",
}
SENTIMENT_ORDER = ["negatif", "netral", "positif"]


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\bnya\b", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(tokens):
    return [NORMALIZED_WORD_DICT.get(t, t) for t in tokens]


def remove_stopwords(tokens, sw_set):
    return [w for w in tokens if w not in sw_set]


@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


@st.cache_data(show_spinner="⏳ Memproses data…")
def process_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline identical to the notebook."""
    df = raw_df.copy()

    # ── rename ──────────────────────────────────────────────────────────
    df = df.rename(columns={
        "_ap3a": "username",
        "x1lliihq 2": "text",
        "x1lliihq 3": "likes",
    })
    df = df[["username", "text", "likes"]]
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    # ── clean ───────────────────────────────────────────────────────────
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.strip() != ""]
    df = df[df["clean"].str.split().str.len() >= 2]
    df = df.reset_index(drop=True)

    # ── tokenise ────────────────────────────────────────────────────────
    regexp = RegexpTokenizer(r"\w+")
    df["token"] = df["clean"].apply(regexp.tokenize)

    # ── normalise ───────────────────────────────────────────────────────
    df["normalisasi"] = df["token"].apply(normalize_tokens)

    # ── stopwords ───────────────────────────────────────────────────────
    sw_set = set(nltk_stopwords.words("indonesian")).union(ADDITIONAL_STOPWORDS)
    df["stopwords"] = df["normalisasi"].apply(lambda t: remove_stopwords(t, sw_set))

    # ── stemming ────────────────────────────────────────────────────────
    stemmer = get_stemmer()
    term_dict: dict[str, str] = {}
    for doc in df["stopwords"]:
        for term in doc:
            if term not in term_dict:
                term_dict[term] = stemmer.stem(term)
    df["stemmer"] = df["stopwords"].apply(lambda doc: [term_dict[t] for t in doc])
    df["final"] = df["stemmer"].apply(lambda x: " ".join(x))

    # ── sentiment labelling (InSet Lexicon) ─────────────────────────────
    pos_path, neg_path = "positive.tsv", "negative.tsv"
    url_pos = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    url_neg = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    if not os.path.exists(pos_path):
        urllib.request.urlretrieve(url_pos, pos_path)
    if not os.path.exists(neg_path):
        urllib.request.urlretrieve(url_neg, neg_path)

    df_pos = pd.read_csv(pos_path, sep="\t", dtype={"weight": int})
    df_neg = pd.read_csv(neg_path, sep="\t", dtype={"weight": int})
    pos_dict = dict(zip(df_pos["word"], df_pos["weight"]))
    neg_dict = dict(zip(df_neg["word"], df_neg["weight"]))

    def label_sentiment(text):
        words = str(text).split()
        score = 0
        for w in words:
            if w in pos_dict:
                score += pos_dict[w]
            if w in neg_dict:
                score -= abs(neg_dict[w])
        if score > 0:
            return "positif"
        elif score < 0:
            return "negatif"
        return "netral"

    df["sentimen"] = df["final"].apply(label_sentiment)
    df["length"] = df["stemmer"].apply(len)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=60, b=40),
)


def sentiment_bar(df: pd.DataFrame):
    counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    fig = go.Figure(
        go.Bar(
            x=counts.index.str.capitalize(),
            y=counts.values,
            marker_color=[SENTIMENT_COLORS[s] for s in counts.index],
            text=counts.values,
            textposition="outside",
            textfont=dict(size=14, weight="bold"),
        )
    )
    fig.update_layout(
        title=dict(text="Distribusi Label Sentimen (3 Kelas)", font=dict(size=18)),
        xaxis_title="Sentimen",
        yaxis_title="Jumlah",
        **PLOTLY_LAYOUT,
    )
    fig.update_yaxes(gridcolor="rgba(200,200,200,.15)")
    return fig


def sentiment_pie(df: pd.DataFrame):
    counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    fig = go.Figure(
        go.Pie(
            labels=counts.index.str.capitalize(),
            values=counts.values,
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in counts.index]),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=13),
            pull=[0.03] * len(counts),
        )
    )
    fig.update_layout(
        title=dict(text="Proporsi Sentimen", font=dict(size=18)),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def word_freq_bar(df: pd.DataFrame, top_n: int = 20):
    all_words = []
    for tokens in df["token"]:
        all_words.extend(tokens)
    common = Counter(all_words).most_common(top_n)
    words, freqs = zip(*common) if common else ([], [])

    fig = go.Figure(
        go.Bar(
            y=list(reversed(list(words))),
            x=list(reversed(list(freqs))),
            orientation="h",
            marker=dict(
                color=list(reversed(list(freqs))),
                colorscale="Viridis",
                showscale=False,
            ),
            text=list(reversed(list(freqs))),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=f"Top {top_n} Kata Paling Sering Muncul", font=dict(size=18)),
        xaxis_title="Frekuensi",
        yaxis_title="Kata",
        height=max(450, top_n * 28),
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(gridcolor="rgba(200,200,200,.15)")
    return fig


def make_wordcloud(df: pd.DataFrame, sentiment: str | None = None):
    if sentiment:
        subset = df[df["sentimen"] == sentiment]
    else:
        subset = df
    text = " ".join(subset["final"].dropna().tolist())
    if not text.strip():
        return None

    color_map = {
        "positif": "YlGn",
        "negatif": "OrRd",
        "netral": "Blues",
    }
    cmap = color_map.get(sentiment, "viridis")

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=cmap,
        max_words=120,
        max_font_size=120,
        random_state=42,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def text_length_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="length",
        color="sentimen",
        color_discrete_map={s.capitalize(): c for s, c in SENTIMENT_COLORS.items()} | SENTIMENT_COLORS,
        barmode="overlay",
        nbins=40,
        opacity=0.7,
        labels={"length": "Jumlah Kata", "sentimen": "Sentimen"},
        title="Distribusi Panjang Teks per Sentimen",
        category_orders={"sentimen": SENTIMENT_ORDER},
    )
    fig.update_layout(**PLOTLY_LAYOUT, title_font_size=18)
    fig.update_xaxes(gridcolor="rgba(200,200,200,.15)")
    fig.update_yaxes(gridcolor="rgba(200,200,200,.15)")
    return fig


def text_length_box(df: pd.DataFrame):
    fig = px.box(
        df,
        x="sentimen",
        y="length",
        color="sentimen",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"length": "Jumlah Kata", "sentimen": "Sentimen"},
        title="Boxplot Panjang Teks per Sentimen",
        category_orders={"sentimen": SENTIMENT_ORDER},
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, title_font_size=18)
    fig.update_yaxes(gridcolor="rgba(200,200,200,.15)")
    return fig


def ngram_bar(df: pd.DataFrame, n: int = 2, top_k: int = 15):
    from nltk.util import ngrams as nltk_ngrams
    all_ngrams = []
    for tokens in df["stemmer"]:
        all_ngrams.extend(list(nltk_ngrams(tokens, n)))
    common = Counter(all_ngrams).most_common(top_k)
    if not common:
        return None
    labels = [" ".join(g) for g, _ in reversed(common)]
    values = [c for _, c in reversed(common)]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=values, colorscale="Plasma", showscale=False),
            text=values,
            textposition="outside",
        )
    )
    name = {2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram")
    fig.update_layout(
        title=dict(text=f"Top {top_k} {name}", font=dict(size=18)),
        xaxis_title="Frekuensi",
        height=max(420, top_k * 30),
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(gridcolor="rgba(200,200,200,.15)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Navigasi")
    page = st.radio(
        "Pilih halaman",
        ["🏠 Overview", "📈 Exploratory Data Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
        <div style='text-align:center; opacity:.6; font-size:.78rem; margin-top:1rem;'>
            <b>Project UAS ADTT – Kelompok 8</b><br>
            Enggar Respati · M. Roihan A.S.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_data() -> pd.DataFrame | None:
    """Try local file first, then show uploader."""
    local_path = os.path.join(os.path.dirname(__file__), "reviews_rp.csv")
    if os.path.exists(local_path):
        raw = load_csv(local_path)
        return process_data(raw)

    st.info("📂 File `reviews_rp.csv` tidak ditemukan di direktori dashboard. Silakan upload di bawah.")
    uploaded = st.file_uploader("Upload reviews_rp.csv", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        return process_data(raw)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame):
    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Dashboard Analisis Sentimen</h1>
            <p>Komentar Instagram · Analisis Teks & Deteksi Sentimen dengan NLP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards ────────────────────────────────────────────────────────
    counts = df["sentimen"].value_counts()
    total = len(df)
    neg = int(counts.get("negatif", 0))
    net = int(counts.get("netral", 0))
    pos = int(counts.get("positif", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Komentar", f"{total:,}")
    c2.metric("😠 Negatif", f"{neg:,}", delta=f"{neg/total*100:.1f}%", delta_color="inverse")
    c3.metric("😐 Netral", f"{net:,}", delta=f"{net/total*100:.1f}%", delta_color="off")
    c4.metric("😊 Positif", f"{pos:,}", delta=f"{pos/total*100:.1f}%")

    st.write("")

    # ── Charts side by side ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(sentiment_bar(df), use_container_width=True)
    with col_b:
        st.plotly_chart(sentiment_pie(df), use_container_width=True)

    # ── Sample data ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Sampel Data</p>', unsafe_allow_html=True)
    display_cols = ["username", "text", "sentimen"]
    st.dataframe(
        df[display_cols].head(10).style.applymap(
            lambda v: f"color: {SENTIMENT_COLORS.get(v, 'inherit')}; font-weight:600",
            subset=["sentimen"],
        ),
        use_container_width=True,
        height=400,
    )


def page_eda(df: pd.DataFrame):
    st.markdown(
        """
        <div class="hero">
            <h1>📈 Exploratory Data Analysis</h1>
            <p>Visualisasi mendalam terhadap data komentar & sentimen</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📊 Distribusi Sentimen",
        "💬 Frekuensi Kata",
        "☁️ Word Cloud",
        "📏 Panjang Teks",
        "🔗 N-gram",
    ])

    # ── Tab 1: Distribusi Sentimen ───────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sentiment_bar(df), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)

        # Table
        st.markdown('<p class="section-header">📋 Detail Distribusi</p>', unsafe_allow_html=True)
        counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
        pct = (counts / counts.sum() * 100).round(2)
        dist_df = pd.DataFrame({"Sentimen": counts.index.str.capitalize(), "Jumlah": counts.values, "Persentase (%)": pct.values})
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Frekuensi Kata ────────────────────────────────────────────
    with tabs[1]:
        top_n = st.slider("Jumlah kata teratas", 10, 40, 20, key="freq_slider")
        st.plotly_chart(word_freq_bar(df, top_n), use_container_width=True)

    # ── Tab 3: Word Cloud ────────────────────────────────────────────────
    with tabs[2]:
        wc_option = st.radio(
            "Pilih sentimen",
            ["Semua", "Positif", "Negatif", "Netral"],
            horizontal=True,
            key="wc_radio",
        )
        sentiment_key = None if wc_option == "Semua" else wc_option.lower()
        fig_wc = make_wordcloud(df, sentiment_key)
        if fig_wc:
            st.pyplot(fig_wc, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

    # ── Tab 4: Panjang Teks ──────────────────────────────────────────────
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(text_length_histogram(df), use_container_width=True)
        with c2:
            st.plotly_chart(text_length_box(df), use_container_width=True)

        # descriptive stats
        st.markdown('<p class="section-header">📊 Statistik Deskriptif (Panjang Teks)</p>', unsafe_allow_html=True)
        stats = df.groupby("sentimen")["length"].describe().round(2)
        st.dataframe(stats, use_container_width=True)

    # ── Tab 5: N-gram ────────────────────────────────────────────────────
    with tabs[4]:
        nc1, nc2 = st.columns(2)
        with nc1:
            fig_bi = ngram_bar(df, n=2, top_k=15)
            if fig_bi:
                st.plotly_chart(fig_bi, use_container_width=True)
        with nc2:
            fig_tri = ngram_bar(df, n=3, top_k=15)
            if fig_tri:
                st.plotly_chart(fig_tri, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = get_data()
    if df is None:
        st.stop()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📈 Exploratory Data Analysis":
        page_eda(df)


if __name__ == "__main__":
    main()
