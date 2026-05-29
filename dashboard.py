import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        transition: all .2s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #a78bfa !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,.10) 0%, rgba(168,85,247,.08) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(139,92,246,.20);
        border-radius: 20px;
        padding: 22px 26px;
        box-shadow: 0 8px 32px rgba(0,0,0,.06), inset 0 1px 0 rgba(255,255,255,.1);
        transition: all .3s cubic-bezier(.4,0,.2,1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 48px rgba(99,102,241,.20), inset 0 1px 0 rgba(255,255,255,.15);
        border-color: rgba(139,92,246,.4);
    }
    div[data-testid="stMetric"] label {
        font-size: .78rem !important;
        letter-spacing: .06em;
        text-transform: uppercase;
        opacity: .65;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 2rem 0 1rem;
        padding-bottom: .5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        display: inline-block;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 12px 40px rgba(102,126,234,.35);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,.08) 0%, transparent 60%);
        animation: hero-shimmer 8s ease-in-out infinite;
    }
    @keyframes hero-shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(10%, 10%); }
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 900;
        margin: 0;
        position: relative;
        text-shadow: 0 2px 12px rgba(0,0,0,.15);
    }
    .hero p {
        font-size: 1.1rem;
        opacity: .92;
        margin-top: .6rem;
        position: relative;
        font-weight: 400;
    }

    /* ── Plotly chart containers ── */
    .stPlotlyChart {
        border-radius: 18px;
        overflow: hidden;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(99,102,241,.06);
        border-radius: 14px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: .9rem;
        transition: all .2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(139,92,246,.15);
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
    "netral":  "#a78bfa",
    "positif": "#10b981",
}
SENTIMENT_ORDER = ["negatif", "netral", "positif"]
SENTIMENT_LABELS = {"negatif": "😠 Negatif", "netral": "😐 Netral", "positif": "😊 Positif"}


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convert a hex color string to an rgba() string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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

    # ── rename
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

    # ── clean
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.strip() != ""]
    df = df[df["clean"].str.split().str.len() >= 2]
    df = df.reset_index(drop=True)

    # ── tokenise
    regexp = RegexpTokenizer(r"\w+")
    df["token"] = df["clean"].apply(regexp.tokenize)

    # ── normalise
    df["normalisasi"] = df["token"].apply(normalize_tokens)

    # ── stopwords
    sw_set = set(nltk_stopwords.words("indonesian")).union(ADDITIONAL_STOPWORDS)
    df["stopwords"] = df["normalisasi"].apply(lambda t: remove_stopwords(t, sw_set))

    # ── stemming
    stemmer = get_stemmer()
    term_dict = {}
    for doc in df["stopwords"]:
        for term in doc:
            if term not in term_dict:
                term_dict[term] = stemmer.stem(term)
    df["stemmer"] = df["stopwords"].apply(lambda doc: [term_dict[t] for t in doc])
    df["final"] = df["stemmer"].apply(lambda x: " ".join(x))

    # ── sentiment labelling (InSet Lexicon)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_path = os.path.join(script_dir, "positive.tsv")
    neg_path = os.path.join(script_dir, "negative.tsv")
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
#  CHART HELPERS — PREMIUM VISUALS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#374151"),
    margin=dict(l=48, r=48, t=72, b=48),
    hoverlabel=dict(
        bgcolor="rgba(255,255,255,0.95)",
        font=dict(size=13, family="Inter, sans-serif"),
        bordercolor="rgba(0,0,0,0.1)",
    ),
)


def sentiment_bar(df):
    counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    total = counts.sum()

    fig = go.Figure()
    for sent, val in counts.items():
        pct = val / total * 100
        fig.add_trace(go.Bar(
            x=[SENTIMENT_LABELS[sent]],
            y=[val],
            marker=dict(
                color=SENTIMENT_COLORS[sent],
                line=dict(width=0),
            ),
            text=[f"<b>{val:,}</b><br><span style='font-size:11px;opacity:.7'>{pct:.1f}%</span>"],
            textposition="outside",
            textfont=dict(size=14),
            hovertemplate=(
                f"<b>{SENTIMENT_LABELS[sent]}</b><br>"
                f"Jumlah: {val:,}<br>"
                f"Persentase: {pct:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Distribusi Label Sentimen</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="", tickfont=dict(size=13)),
        yaxis=dict(
            title=dict(text="Jumlah Komentar", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.2)",
            gridwidth=1,
            zeroline=False,
        ),
        bargap=0.35,
        height=420,
        **PLOTLY_LAYOUT,
    )
    return fig


def sentiment_pie(df):
    counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    labels = [SENTIMENT_LABELS[s] for s in counts.index]
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        marker=dict(
            colors=colors,
            line=dict(color="white", width=3),
        ),
        hole=0.55,
        textinfo="percent",
        textfont=dict(size=15, color="white"),
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persentase: %{percent}<extra></extra>",
        pull=[0.04, 0.02, 0.04],
        rotation=45,
    ))

    # Center annotation
    total = counts.sum()
    fig.add_annotation(
        text=(
            f"<b style='font-size:28px;color:#1f2937'>{total:,}</b><br>"
            f"<span style='font-size:11px;color:#6b7280'>Total</span>"
        ),
        showarrow=False, font=dict(size=14),
    )

    fig.update_layout(
        title=dict(
            text="<b>Proporsi Sentimen</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        height=420,
        **PLOTLY_LAYOUT,
    )
    return fig


def word_freq_bar(df, top_n=20):
    all_words = []
    for tokens in df["token"]:
        all_words.extend(tokens)
    common = Counter(all_words).most_common(top_n)
    if not common:
        return go.Figure()
    words, freqs = zip(*common)
    words = list(reversed(words))
    freqs = list(reversed(list(freqs)))
    max_f = max(freqs) if freqs else 1

    # Generate gradient colors from cool blue to warm purple
    colors = [
        f"rgba({int(102 + (153 * f / max_f))}, {int(126 - (40 * f / max_f))}, {int(234 - (60 * f / max_f))}, 0.85)"
        for f in freqs
    ]

    fig = go.Figure(go.Bar(
        y=words,
        x=freqs,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"<b>{f:,}</b>" for f in freqs],
        textposition="outside",
        textfont=dict(size=12, color="#4b5563"),
        hovertemplate="<b>%{y}</b><br>Frekuensi: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Top {top_n} Kata Paling Sering Muncul</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Frekuensi", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.2)",
            zeroline=False,
        ),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=max(500, top_n * 32),
        **PLOTLY_LAYOUT,
    )
    return fig


def _wc_color_positif(*args, **kwargs):
    return np.random.choice(["#059669", "#10b981", "#34d399", "#065f46", "#047857"])

def _wc_color_negatif(*args, **kwargs):
    return np.random.choice(["#dc2626", "#ef4444", "#f87171", "#991b1b", "#b91c1c"])

def _wc_color_netral(*args, **kwargs):
    return np.random.choice(["#7c3aed", "#8b5cf6", "#a78bfa", "#6d28d9", "#5b21b6"])

def _wc_color_default(*args, **kwargs):
    return np.random.choice([
        "#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b",
        "#fa709a", "#fee140", "#30cfd0", "#a18cd1",
    ])

_WC_COLOR_FUNCS = {
    "positif": _wc_color_positif,
    "negatif": _wc_color_negatif,
    "netral": _wc_color_netral,
}


def make_wordcloud(df, sentiment=None):
    if sentiment:
        subset = df[df["sentimen"] == sentiment]
    else:
        subset = df
    text = " ".join(subset["final"].dropna().tolist())
    if not text.strip():
        return None

    color_func = _WC_COLOR_FUNCS.get(sentiment, _wc_color_default)

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=150,
        max_font_size=140,
        min_font_size=12,
        random_state=42,
        collocations=False,
        prefer_horizontal=0.8,
        relative_scaling=0.5,
        color_func=color_func,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def text_length_histogram(df):
    fig = go.Figure()
    for sent in SENTIMENT_ORDER:
        sub = df[df["sentimen"] == sent]
        fig.add_trace(go.Histogram(
            x=sub["length"],
            name=SENTIMENT_LABELS[sent],
            marker=dict(
                color=_hex_to_rgba(SENTIMENT_COLORS[sent], 0.7),
                line=dict(color="white", width=1),
            ),
            nbinsx=35,
            hovertemplate="Panjang: %{x}<br>Jumlah: %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(
            text="<b>Distribusi Panjang Teks per Sentimen</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        barmode="overlay",
        xaxis=dict(
            title=dict(text="Jumlah Kata", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.15)",
        ),
        yaxis=dict(
            title=dict(text="Frekuensi", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.15)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        height=420,
        **PLOTLY_LAYOUT,
    )
    return fig


def text_length_box(df):
    fig = go.Figure()
    for sent in SENTIMENT_ORDER:
        sub = df[df["sentimen"] == sent]
        fig.add_trace(go.Box(
            y=sub["length"],
            name=SENTIMENT_LABELS[sent],
            marker=dict(
                color=SENTIMENT_COLORS[sent],
                outliercolor=SENTIMENT_COLORS[sent],
                size=4,
            ),
            line=dict(color=SENTIMENT_COLORS[sent], width=2),
            fillcolor=_hex_to_rgba(SENTIMENT_COLORS[sent], 0.15),
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="Sentimen: %{x}<br>Panjang: %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(
            text="<b>Boxplot Panjang Teks per Sentimen</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        showlegend=False,
        xaxis=dict(title="", tickfont=dict(size=13)),
        yaxis=dict(
            title=dict(text="Jumlah Kata", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.15)",
            zeroline=False,
        ),
        height=420,
        **PLOTLY_LAYOUT,
    )
    return fig


def ngram_bar(df, n=2, top_k=15):
    from nltk.util import ngrams as nltk_ngrams
    all_ngrams = []
    for tokens in df["stemmer"]:
        all_ngrams.extend(list(nltk_ngrams(tokens, n)))
    common = Counter(all_ngrams).most_common(top_k)
    if not common:
        return None
    labels = [" ".join(g) for g, _ in reversed(common)]
    values = [c for _, c in reversed(common)]
    max_v = max(values) if values else 1

    # Gradient from teal to violet
    colors = [
        f"hsl({int(260 - (120 * v / max_v))}, 70%, {int(50 + (15 * (1 - v / max_v)))}%)"
        for v in values
    ]

    name = {2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram")
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"<b>{v}</b>" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#4b5563"),
        hovertemplate="<b>%{y}</b><br>Frekuensi: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>Top {top_k} {name}</b>",
            font=dict(size=20, color="#1f2937"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Frekuensi", font=dict(size=12)),
            gridcolor="rgba(200,200,220,.15)",
            zeroline=False,
        ),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=max(460, top_k * 34),
        **PLOTLY_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 .5rem;">
            <span style="font-size:2.5rem;">📊</span>
            <h2 style="margin:.4rem 0 0; font-size:1.15rem; font-weight:800;
                        background: linear-gradient(135deg, #a78bfa, #f093fb);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Navigasi Dashboard
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Pilih halaman",
        ["🏠 Overview", "📈 Exploratory Data Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
        <div style='text-align:center; opacity:.5; font-size:.75rem; margin-top:2rem; line-height:1.6;'>
            <b>Project UAS ADTT</b><br>
            Kelompok 8<br><br>
            <span style="opacity:.7">Enggar Respati<br>M. Roihan A.S.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


def get_data():
    """Try local file first, then show uploader."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "reviews_rp.csv")
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

def page_overview(df):
    # ── Hero
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Dashboard Analisis Sentimen</h1>
            <p>Komentar Instagram · Analisis Teks &amp; Deteksi Sentimen dengan NLP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards
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

    # ── Charts side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(sentiment_bar(df), use_container_width=True)
    with col_b:
        st.plotly_chart(sentiment_pie(df), use_container_width=True)

    # ── Sample data
    st.markdown('<p class="section-header">📋 Sampel Data</p>', unsafe_allow_html=True)
    display_cols = ["username", "text", "sentimen"]
    st.dataframe(
        df[display_cols].head(10).style.map(
            lambda v: f"color: {SENTIMENT_COLORS.get(v, 'inherit')}; font-weight:600",
            subset=["sentimen"],
        ),
        use_container_width=True,
        height=400,
    )


def page_eda(df):
    st.markdown(
        """
        <div class="hero">
            <h1>📈 Exploratory Data Analysis</h1>
            <p>Visualisasi mendalam terhadap data komentar &amp; sentimen</p>
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

    # ── Tab 1: Distribusi Sentimen
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sentiment_bar(df), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)

        st.markdown('<p class="section-header">📋 Detail Distribusi</p>', unsafe_allow_html=True)
        dist_counts = df["sentimen"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
        pct = (dist_counts / dist_counts.sum() * 100).round(2)
        dist_df = pd.DataFrame({
            "Sentimen": [SENTIMENT_LABELS[s] for s in dist_counts.index],
            "Jumlah": dist_counts.values,
            "Persentase (%)": pct.values,
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Frekuensi Kata
    with tabs[1]:
        top_n = st.slider("Jumlah kata teratas", 10, 40, 20, key="freq_slider")
        st.plotly_chart(word_freq_bar(df, top_n), use_container_width=True)

    # ── Tab 3: Word Cloud
    with tabs[2]:
        wc_option = st.radio(
            "Pilih sentimen untuk Word Cloud",
            ["Semua", "Positif", "Negatif", "Netral"],
            horizontal=True,
            key="wc_radio",
        )

        sentiment_key = None if wc_option == "Semua" else wc_option.lower()

        st.markdown(
            f"<p style='text-align:center; font-size:1.1rem; font-weight:700; margin:1rem 0;'>"
            f"☁️ Word Cloud — <span style='color:{SENTIMENT_COLORS.get(sentiment_key, '#667eea')}'>"
            f"{wc_option}</span></p>",
            unsafe_allow_html=True,
        )

        fig_wc = make_wordcloud(df, sentiment_key)
        if fig_wc:
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

    # ── Tab 4: Panjang Teks
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(text_length_histogram(df), use_container_width=True)
        with c2:
            st.plotly_chart(text_length_box(df), use_container_width=True)

        st.markdown('<p class="section-header">📊 Statistik Deskriptif (Panjang Teks)</p>', unsafe_allow_html=True)
        stats = df.groupby("sentimen")["length"].describe().round(2)
        stats.index = [SENTIMENT_LABELS.get(s, s) for s in stats.index]
        st.dataframe(stats, use_container_width=True)

    # ── Tab 5: N-gram
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
