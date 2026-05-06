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
from typing import Dict, List, Tuple

try:
    from spellchecker import SpellChecker
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspellchecker", "-q"])
    from spellchecker import SpellChecker
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
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
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

# --- Step 7: Universal stopwords ---
NEUTRAL_STOPS = {
    # Fragrance-specific neutrals
    "smell", "smells", "smelling", "scent", "scents", "odor", "odour",
    "perfume", "fragrance", "aroma", "note", "nuance", "touch", "hint", "product",
    # Sensory / perception fillers
    "feel", "feels", "feeling", "feelings", "felt",
    "impression", "image", "association", "associates", "reminds",
    "remind", "remember", "evoke", "seem", "find",
    # Generic filler verbs
    "think", "make", "let", "go", "put", "get", "give", "come", "take",
    "look", "say", "use", "suit", "mind",
    # Connectors / discourse markers (previously split clauses — now just removed)
    "also", "though", "simply", "somehow", "therefore", "order", "almost", "just",
    "but", "however", "although", "yet", "whereas", "nevertheless", "nonetheless",
    "despite", "still", "even", "except", "besides", "otherwise", "instead",
    # Quantity / degree fillers
    "little", "lot", "bit", "quite", "really", "more", "less", "enough",
    "most", "few", "many", "some", "any", "all", "both", "either",
    "neither", "one", "two", "each", "same", "other", "own",
    # Generic qualifiers
    "like", "kind", "kinda", "real", "something", "anything", "everything", "nothing",
    # Time / context fillers
    "day", "days", "time", "times", "moment", "moments",
    "season", "year", "week",
    # Generic descriptors with no olfactive value
    "color", "colour", "size", "shape", "texture", "weight", "price", "cost",
    "room", "place", "area", "space", "world",
    "dry", "wet",
    "first", "last", "next", "every", "much", "well", "right",
    "good", "bad", "great", "nice",
    # Pronouns
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    # Auxiliary / modal verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "could", "should", "may", "might", "shall", "can", "cannot", "ought",
    # Prepositions / articles / conjunctions
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with", "by", "as",
    "from", "into", "about", "above", "after", "again", "against",
    "before", "below", "between", "down", "during", "off", "once",
    "only", "out", "over", "through", "under", "up", "while",
    "and", "or", "if", "that", "this", "these", "those",
    "there", "here", "than", "then", "such", "so",
    # Question words
    "who", "whom", "which", "when", "where", "why", "how", "what",
    # Misc
    "because", "until", "further", "too", "very",
    "therefore", "hence", "thus", "already",
    "indeed", "perhaps", "maybe",
    # Adverbs / filler opinions with no olfactive value
    "upon", "always", "slightly", "found", "try", "afterward", "afterwards",
    "opinion", "overall", "generally", "usually", "often", "sometimes",
    "never", "ever", "rather", "quite", "fairly", "nearly", "almost",
    "mostly", "mainly", "largely", "particularly", "especially", "specifically",
    "certain", "certainly", "definitely", "absolutely", "completely", "totally",
    "actually", "basically", "essentially", "literally", "personally",
    "immediately", "quickly", "slowly", "suddenly", "finally", "eventually",
    "initially", "originally", "previously", "recently", "currently",
    "anyway", "anyhow", "somehow", "somewhere", "sometime", "sometimes",
    "seem", "seems", "seemed", "appear", "appears", "appeared",
    "become", "became", "becomes", "keep", "kept", "keeps",
    "want", "wanted", "wants", "need", "needed", "needs",
    "try", "tried", "tries", "start", "started", "begin", "began",
    "way", "ways", "thing", "things", "stuff", "bit", "bits",
    "lot", "lots", "number", "amount", "kind", "type", "sort",
    "big", "small", "long", "short", "high", "low",
    "same", "whole", "entire", "full",
    "able", "unable", "likely", "unlikely", "possible", "impossible",
    "know", "knew", "known", "think", "thought", "feel", "felt",
    "mean", "meant", "means", "show", "showed", "shown",
    "tell", "told", "told", "ask", "asked", "answer", "answered",
    "plus", "minus", "around", "along", "across", "within", "without",
    "since", "although", "though", "whereas", "while", "whenever",
    "whether", "whatever", "wherever", "whoever", "however", "whichever",
    "nobody", "nothing", "nowhere", "someone", "somewhere", "something",
    "anyone", "anywhere", "everyone", "everywhere", "everything",
    "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "mr", "mrs", "ms", "dr", "etc", "ie", "eg","open","bottle","opening","wet","dry"
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

def _init_spellchecker(protect_words: frozenset) -> SpellChecker:
    """Initialise spell checker with protected vocabulary.
    Called once after SPELL_PROTECT is defined.
    """
    sc = SpellChecker()
    sc.word_frequency.load_words(protect_words)
    return sc

SPELL_PROTECT: set = {
    # fragrance families & descriptors
    "woody", "musky", "ozonic", "aquatic", "gourmand", "chypre", "fougere",
    "aldehyde", "aldehydic", "animalic", "balsamic", "camphoric", "lactonic",
    "resinous", "powdery", "smoky", "earthy", "mossy", "rooty",
    "soapy", "cologney", "citrusy", "spicy", "watery", "creamy", "boozy",
    "florals", "woodsy", "freshness",
    # specific ingredients
    "vetiver", "patchouli", "oud", "ambergris", "labdanum", "benzoin",
    "civet", "oakmoss", "bergamot", "neroli", "petitgrain", "ylang",
    "tonka", "orris", "iris", "mimosa", "tuberose", "narcissus",
    "hedione", "galaxolide", "ambroxan", "cashmeran",
    "sandalwood", "cedarwood", "rosewood", "agarwood",
    "jasmine", "gardenia", "magnolia", "wisteria", "peony",
    # negation & intensity tokens — pipeline handles these
    "not", "never", "very", "so", "really", "extremely",
    "incredibly", "super", "highly", "deeply", "absolutely",
    "totally", "quite", "pretty", "awfully", "terribly", "remarkably",
}

@st.cache_resource(show_spinner=False)
def _get_spell():
    return _init_spellchecker(frozenset(SPELL_PROTECT))

# Simple dict cache so identical misspelled tokens are only corrected once
_correction_cache: Dict[str, str] = {}

def _cached_correction(tok: str) -> str:
    if tok not in _correction_cache:
        result = _get_spell().correction(tok)
        _correction_cache[tok] = result if result else tok
    return _correction_cache[tok]

def step3_5_spell_correct(tokens: List[str]) -> List[str]:
    """Correct misspelled tokens using pyspellchecker.

    - Tokens in SPELL_PROTECT are passed through unchanged.
    - Tokens containing underscores (compound tokens) are skipped.
    - Single-character tokens are skipped.
    - Only tokens flagged as unknown are corrected.
    - If no correction candidate is found, the original token is kept.
    """
    candidates = [t for t in tokens if t not in SPELL_PROTECT and "_" not in t and len(t) > 1]
    if not candidates:
        return tokens
    unknown = _get_spell().unknown(candidates)
    if not unknown:
        return tokens
    out = []
    for tok in tokens:
        if tok in SPELL_PROTECT or "_" in tok or len(tok) <= 1 or tok not in unknown:
            out.append(tok)
        else:
            out.append(_cached_correction(tok))
    return out

CATEGORY_STOPS: Dict[str, set] = {
    "fine_fragrance": {
        "perfume", "fragrance", "scent", "spray", "bottle", "vial", "atomizer",
        "apply", "spritz", "spraying", "wrist", "neck", "wear", "wearing", "wore",
        "longevity", "sillage", "projection", "trail", "top", "heart", "base",
        "accord", "nose", "house", "brand", "collection", "launch", 
    },
    "laundry_detergent": {
        "laundry", "wash", "washing", "rinse", "spin", "cycle", "machine",
        "clothes", "clothing", "garment", "garments", "fabric", "fabrics", "linen",
        "detergent", "powder", "liquid", "capsule", "pod", "dose",
        "stain", "stains", "dirt", "dirty", "soil", "soiling","release","bottle","dry","open","opening","wet"
    },
    "fabric_softener": {
        "softener", "conditioner", "soften", "softer",
        "laundry", "wash", "washing", "clothes", "fabric", "linen",
        "rinse", "cycle", "machine", "drying", "static", "wrinkle", "bottle","dry","open","opening","wet"
    },
    "dishwashing": {
        "dish", "dishes", "washing", "wash", "rinse",
        "plate", "plates", "glass", "glasses", "bowl", "cutlery", "pan", "pot",
        "grease", "greasy", "residue", "tablet", "capsule", "pod",
        "foam", "lather", "bubble", "bubbles", "streak", "shine",
    },
    "surface_cleaner": {
        "surface", "surfaces", "countertop", "floor", "tile",
        "kitchen", "bathroom", "toilet", "wipe", "wiping", "spray",
        "scrub", "rinse", "bucket", "bacteria", "germ", "germs",
        "disinfect", "disinfectant", "mold", "mildew", "limescale", "grease",
        "clean", "cleaning", "cleaner", "bleach", "chlorine",
    },
    "body_care": {
        "shower", "bath", "body", "skin", "lather", "rinse", "wash",
        "gel", "lotion", "cream", "moisturize", "moisturizing", "moisturizer",
        "absorb", "absorption", "texture", "consistency",
        "apply", "rub", "massage", "dryness", "oily",
    },
    "hair_care": {
        "hair", "shampoo", "scalp", "strand", "strands", "wash", "washing",
        "rinse", "lather", "drying", "volume", "frizz", "damage", "damaged",
        "repair", "strengthen", "dye", "bleach", "greasy", "dandruff",
        "itch", "smooth", "silky",
    },
    "deodorant": {
        "deodorant", "deo", "antiperspirant", "spray", "roll-on", "stick",
        "underarm", "armpit", "sweat", "sweating", "perspiration",
        "protect", "protection", "hour", "hours", "day", "apply", "dry", "skin",
        "irritation", "sensitive",
    },
    "oral_care": {
        "tooth", "teeth", "toothpaste", "brush", "brushing", "rinse", "spit",
        "cavity", "plaque", "tartar", "whitening", "enamel",
        "gum", "gums", "breath", "clean", "cleaning", "foam", "mint",
    },
    "air_care": {
        "air", "room", "home", "office", "space",
        "spray", "diffuser", "candle", "plug-in", "wick",
        "burn", "melt", "wax", "neutralize", "eliminate", "mask",
        "refresh", "last", "lasting", "hour", "hours",
    },
    "baby_care": {
        "baby", "infant", "child", "toddler",
        "diaper", "wipe", "wipes", "bath", "wash",
        "lotion", "cream", "powder", "hypoallergenic", "skin", "rash",
    },
    "skincare": {
        "skin", "face", "facial", "serum", "cream", "moisturizer",
        "lotion", "gel", "toner", "essence", "apply", "absorb",
        "absorption", "texture", "consistency", "layer", "routine",
        "dryness", "oily", "combination", "sensitive", "pore", "pores",
        "wrinkle", "wrinkles", "aging", "spf", "moisturize", "hydrate",
    },
}

CATEGORY_ALIASES: Dict[str, str] = {
    "fine fragrance":   "fine_fragrance",  "fine_fragrance":  "fine_fragrance",
    "fragrance":        "fine_fragrance",  "perfume":         "fine_fragrance",
    "edp":              "fine_fragrance",  "edt":             "fine_fragrance",
    "cologne":          "fine_fragrance",
    "laundry detergent": "laundry_detergent", "laundry_detergent": "laundry_detergent",
    "fabric care":      "laundry_detergent", "fabric_care":     "laundry_detergent",
    "laundry":          "laundry_detergent", "detergent":       "laundry_detergent",
    "fabric softener":  "fabric_softener", "softener":        "fabric_softener",
    "conditioner":      "fabric_softener",
    "dishwashing":      "dishwashing",     "dish":            "dishwashing",
    "surface cleaner":  "surface_cleaner", "surface_cleaner": "surface_cleaner",
    "body care":        "body_care",       "body_care":       "body_care",
    "shower gel":       "body_care",       "body wash":       "body_care",
    "hair care":        "hair_care",       "hair_care":       "hair_care",
    "shampoo":          "hair_care",
    "deodorant":        "deodorant",       "deo":             "deodorant",
    "oral care":        "oral_care",       "oral_care":       "oral_care",
    "toothpaste":       "oral_care",
    "air care":         "air_care",        "air_care":        "air_care",
    "candle":           "air_care",
    "baby care":        "baby_care",       "baby_care":       "baby_care",
    "baby":             "baby_care",
    "skincare":         "skincare",        "skin care":       "skincare",
    "face cream":       "skincare",
}


def get_category_stops(category: str) -> set:
    """Return the category-specific stopword set for a given category string.
    Returns an empty set if the category is unknown or None.
    """
    if not category or not isinstance(category, str):
        return set()
    key = CATEGORY_ALIASES.get(category.strip().lower())
    if key is None:
        cat_lower = category.strip().lower()
        for alias, canonical in CATEGORY_ALIASES.items():
            if alias in cat_lower or cat_lower in alias:
                key = canonical
                break
    return CATEGORY_STOPS.get(key, set()) if key else set()


WEIGHT_VERY  = 1.5
WEIGHT_PLAIN = 1.0

_PUNCT_STRIP = re.compile(r"[" + re.escape(string.punctuation.replace("-", "")) + r"]")

# =============================================================================
# SESSION STATE INIT — editable settings stored in session state
# (so changes survive Streamlit reruns)
# =============================================================================

if "ss_stops" not in st.session_state:
    st.session_state["ss_stops"] = set(NEUTRAL_STOPS)
if "ss_merges" not in st.session_state:
    st.session_state["ss_merges"] = dict(FRAGRANCE_MERGES)
if "ss_protect" not in st.session_state:
    st.session_state["ss_protect"] = set(PREFIX_STRIP_PROTECT)
if "ss_autocorrect" not in st.session_state:
    st.session_state["ss_autocorrect"] = True
if "ss_category" not in st.session_state:
    st.session_state["ss_category"] = "fine_fragrance"


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
    protect = st.session_state["ss_protect"]
    result = []
    for tok in tokens:
        if tok in protect:
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
    merges = st.session_state["ss_merges"]
    return [merges.get(tok, tok) for tok in tokens]

def step7_stopword_removal(tokens: List[str], cat_stops: set = None) -> List[str]:
    stops = st.session_state["ss_stops"] | (cat_stops or set())
    return [t for t in tokens if t in NEGATION_TERMS or t in INTENSITY_MAP
            or "_" in t or t not in stops]

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


def step10_clean_orphans(tokens: List[str]) -> List[str]:
    """
    Remove standalone negation/intensity tokens that were not collapsed
    into not_X / very_X bigrams — they carry no meaning on their own.
    e.g. ["clean", "not", "fresh"] where "not" couldn't attach to anything.
    Only removes if the token IS a bare negation/intensity word (no underscore).
    """
    orphans = (NEGATION_TERMS | set(INTENSITY_MAP.keys())) - {"no"}
    return [t for t in tokens if t not in orphans]

def process_verbatim(raw_text: str, autocorrect: bool = True,
                     category: str = None) -> List[str]:
    if not raw_text or not isinstance(raw_text, str): return []
    cat_stops = get_category_stops(category) if category else set()
    text = step1_char_normalize(raw_text)
    text = step2_expand_contractions(text)
    tokens = step4_tokenize(text)
    if autocorrect:
        tokens = step3_5_spell_correct(tokens)   # Step 3.5 — spell correction
    tokens = step5_prefix_strip(tokens)
    tokens = step6_lemmatize(tokens)
    tokens = step5_5_synonym_merge(tokens)
    tokens = step_feel_good_bigram(tokens)
    tokens = step7_stopword_removal(tokens, cat_stops)
    tokens = step7_5_intensity_normalize(tokens)
    tokens = step8_ngram_collapse(tokens)
    tokens = step9_dedup(tokens)
    return step10_clean_orphans(tokens)

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


def run_ca(df: pd.DataFrame, p_col: str, fmin: int) -> tuple:
    """
    True Correspondence Analysis on a product × token contingency table.

    Returns
    -------
    (row_coords, col_coords, products, words, inertia_ratio, word_freqs) or (None, error_str)
    """
    grouped = df.groupby(p_col)["token_str"].apply(lambda x: " ".join(x))
    if len(grouped) < 3:
        return None, "Need 3+ products for Correspondence Analysis."

    # Build raw count matrix (products × tokens) — CA requires counts, not TF-IDF
    vec = CountVectorizer(min_df=max(1, fmin), token_pattern=r"(?u)\b\S+\b")
    try:
        X = vec.fit_transform(grouped).toarray().astype(float)
    except ValueError as e:
        return None, str(e)

    words    = vec.get_feature_names_out()
    products = grouped.index.tolist()

    if X.shape[1] < 3:
        return None, "Too few tokens — lower Min Word Frequency."

    # ── True CA via SVD ──────────────────────────────────────────────────
    N  = X.sum()
    P  = X / N                                    # correspondence matrix
    r  = P.sum(axis=1, keepdims=True)             # row masses
    c  = P.sum(axis=0, keepdims=True)             # col masses
    Dr_sqrt_inv = np.diag(1.0 / np.sqrt(r.flatten()))
    Dc_sqrt_inv = np.diag(1.0 / np.sqrt(c.flatten()))
    S  = Dr_sqrt_inv @ (P - r @ c) @ Dc_sqrt_inv  # standardised residuals

    try:
        U, d, Vt = np.linalg.svd(S, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, "SVD failed — try adjusting Min Word Frequency."

    k = min(2, len(d))
    # Principal coordinates (symmetric map — rows and cols comparable)
    row_coords = Dr_sqrt_inv @ U[:, :k] * d[:k]
    col_coords = Dc_sqrt_inv @ Vt[:k, :].T * d[:k]

    inertia     = d**2
    total_iner  = inertia.sum() if inertia.sum() > 0 else 1.0
    inertia_ratio = inertia[:k] / total_iner

    # Per-token total frequency (for sizing)
    word_freqs = np.asarray(X.sum(axis=0)).flatten()

    return (row_coords, col_coords, products, words, inertia_ratio, word_freqs), None


# Keep old name as alias for backward compat
def run_fca(df, p_col, fmin, use_tfidf=False):
    return run_ca(df, p_col, fmin)



# =============================================================================
# ▌BLOCK 3B — SIMILARITY
# =============================================================================

def build_similarity_matrix(df: pd.DataFrame, p_col: str,
                             use_tfidf: bool = True) -> tuple:
    """
    Compute pairwise cosine similarity for all fragrances.

    use_tfidf=True  → TF-IDF (IDF fitted on full corpus)
    use_tfidf=False → Raw token counts (CountVectorizer)
                      Recommended for paired comparisons: word frequency
                      differences are preserved as-is, no IDF distortion.
    """
    grouped = (df.groupby(p_col)["token_str"]
               .apply(lambda x: " ".join(x)))
    frags = grouped.index.tolist()
    docs  = grouped.values.tolist()
    try:
        if use_tfidf:
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\S+\b")
        else:
            vec = CountVectorizer(token_pattern=r"(?u)\b\S+\b")
        mat = vec.fit_transform(docs)
        sim = cosine_similarity(mat)
        return sim, frags
    except Exception:
        return None, []


def stretch_scores(scores: list, low: float = 20.0, high: float = 90.0) -> list:
    """
    Linearly stretch a list of raw scores to [low, high] range.
    Preserves relative ordering while maximising visual spread.

    Special cases:
      - 1 score  → return raw cosine × 100 (no stretch, meaningful absolute value)
      - all same → return midpoint for all
    """
    if len(scores) == 1:
        return [round(scores[0] * 100, 1)]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [round((low + high) / 2, 1)] * len(scores)
    return [round(low + (v - mn) / (mx - mn) * (high - low), 1) for v in scores]




# =============================================================================
# ▌BLOCK 4 — SIDEBAR & FILE LOADING
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

    fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
    use_tfidf   = st.toggle("Use TF-IDF Weighting", value=False)
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

    st.divider()
    autocorrect_on = st.toggle(
        "🔤 Spell Autocorrect",
        value=st.session_state["ss_autocorrect"],
        help="Corrects misspelled words. Fragrance terms (vetiver, oud, etc.) are protected."
    )
    st.session_state["ss_autocorrect"] = autocorrect_on

    cat_options   = ["None"] + list(CATEGORY_STOPS.keys())
    _cur_cat      = st.session_state["ss_category"]
    cat_selection = st.selectbox(
        "Product Category",
        cat_options,
        index=cat_options.index(_cur_cat) if _cur_cat in cat_options else 0,
        help="Removes category-specific filler words on top of universal stopwords."
    )
    st.session_state["ss_category"] = cat_selection

    if uploaded_file and st.button("🚀 Run Analysis"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col]).copy()

        _ac  = st.session_state["ss_autocorrect"]
        _cat = None if st.session_state["ss_category"] == "None" else st.session_state["ss_category"]
        with st.spinner("Running V3 pipeline…"):
            df_filtered["tokens"]    = df_filtered[v_col].apply(
                lambda x: process_verbatim(x, autocorrect=_ac, category=_cat))
            df_filtered["token_str"] = df_filtered["tokens"].apply(tokens_to_string)

        st.session_state["processed_df"] = df_filtered
        st.session_state["filter_info"]  = filter_label
        st.session_state["pref_col"]     = s_col
        st.session_state["p_col"]        = p_col
        st.session_state["v_col"]        = v_col
        st.session_state["ss_autocorrect"] = _ac
        st.session_state["ss_category"]    = cat_selection
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
        st.subheader("⚔️ Similarity to Reference")
        st.caption(
            "IDF fitted on all fragrances. "
            "With 2+ candidates: scores stretched 20–90%. "
            "With 1 candidate: raw cosine (cross-test comparable)."
        )

        ref_p     = st.selectbox("🎯 Reference Fragrance", p_list, index=0)
        cand_list = [p for p in p_list if p != ref_p]
        cand_sel  = st.multiselect(
            "🧪 Candidate(s) to compare",
            cand_list,
            default=cand_list[:min(4, len(cand_list))],
        )

        use_stretch = st.toggle(
            "📊 Stretch scores (20–90%)",
            value=True,
            help="ON: stretched for visual spread within one test. "
                 "OFF: raw cosine — comparable across tests."
        )

        if cand_sel:
            # ── similarity matrix ─────────────────────────────────────────
            sim_matrix, frag_list = build_similarity_matrix(
                df, p_col, use_tfidf=use_tfidf)

            if sim_matrix is None:
                st.error("Could not compute similarity matrix.")
            else:
                ref_idx    = frag_list.index(ref_p)
                raw_scores = [float(sim_matrix[ref_idx, frag_list.index(c)])
                              for c in cand_sel]

                if use_stretch:
                    display_scores = stretch_scores(raw_scores)
                    x_label = "Similarity to reference (stretched 20–90%)"
                else:
                    display_scores = [round(s * 100, 1) for s in raw_scores]
                    x_label = "Similarity to reference (raw cosine %)"

                results = sorted(
                    zip(cand_sel, display_scores, raw_scores),
                    key=lambda x: -x[1]
                )

                # ── Metric cards ──────────────────────────────────────────
                st.divider()
                cols = st.columns(min(len(results), 4))
                for i, (name, score, _) in enumerate(results):
                    cols[i % 4].metric(label=name, value=f"{score}%")

                # ── Bar chart ─────────────────────────────────────────────
                st.divider()
                names_plot  = [r[0] for r in results]
                scores_plot = [r[1] for r in results]
                fig_bar, ax_bar = plt.subplots(figsize=(8, max(2, len(results) * 0.6)))
                colors = plt.cm.RdYlGn([s / 100 for s in scores_plot])
                bars   = ax_bar.barh(names_plot, scores_plot, color=colors)
                ax_bar.set_xlabel(x_label)
                ax_bar.set_xlim(0, 100)
                ax_bar.invert_yaxis()
                for bar, score in zip(bars, scores_plot):
                    ax_bar.text(bar.get_width() + 0.8,
                                bar.get_y() + bar.get_height() / 2,
                                f"{score}%", va="center", fontsize=9)
                ax_bar.set_title(f"Similarity to: {ref_p}", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_bar)

                # ── Token frequency comparison (replaces word clouds) ─────
                st.divider()
                st.write("### 🔤 Token Frequency Breakdown")

                top_n = st.slider("Top N tokens to show", 10, 40, 20, key="tab2_topn")

                for cand in [r[0] for r in results]:
                    st.write(f"---")
                    st.write(f"#### 🎯 {ref_p}  vs  🧪 {cand}")

                    # raw token counts (no IDF)
                    from collections import Counter
                    ref_tokens  = [t for tlist in df[df[p_col].astype(str)==ref_p]["tokens"]
                                   if isinstance(tlist, list) for t in tlist]
                    cand_tokens = [t for tlist in df[df[p_col].astype(str)==cand]["tokens"]
                                   if isinstance(tlist, list) for t in tlist]

                    ref_freq  = Counter(ref_tokens)
                    cand_freq = Counter(cand_tokens)

                    # normalise by total tokens so counts are comparable
                    ref_total  = max(sum(ref_freq.values()), 1)
                    cand_total = max(sum(cand_freq.values()), 1)

                    all_words = set(ref_freq) | set(cand_freq)

                    # shared words — both have freq > 0
                    shared = [(w, ref_freq[w]/ref_total*100,
                                  cand_freq[w]/cand_total*100)
                              for w in all_words
                              if ref_freq[w] > 0 and cand_freq[w] > 0]
                    # sort by average frequency
                    shared.sort(key=lambda x: -(x[1]+x[2])/2)
                    shared = shared[:top_n]

                    # distinctive — only in ref or only in cand
                    ref_only  = [(w, ref_freq[w]/ref_total*100)
                                 for w in all_words
                                 if ref_freq[w] > 0 and cand_freq[w] == 0]
                    cand_only = [(w, cand_freq[w]/cand_total*100)
                                 for w in all_words
                                 if cand_freq[w] > 0 and ref_freq[w] == 0]
                    ref_only.sort(key=lambda x: -x[1])
                    cand_only.sort(key=lambda x: -x[1])
                    ref_only  = ref_only[:top_n]
                    cand_only = cand_only[:top_n]

                    # ── Shared tokens chart ───────────────────────────────
                    if shared:
                        st.write("**🤝 Shared tokens** (both fragrances mention these)")
                        words_s = [w for w,_,_ in shared]
                        vals_r  = [r for _,r,_ in shared]
                        vals_c  = [c for _,_,c in shared]

                        x      = np.arange(len(words_s))
                        width  = 0.35
                        fig_s, ax_s = plt.subplots(figsize=(max(8, len(words_s)*0.55), 4))
                        ax_s.bar(x - width/2, vals_r, width,
                                 label=ref_p[:30],  color="#4C9BE8", alpha=0.85)
                        ax_s.bar(x + width/2, vals_c, width,
                                 label=cand[:30], color="#F28C38", alpha=0.85)
                        ax_s.set_xticks(x)
                        ax_s.set_xticklabels(words_s, rotation=45, ha="right", fontsize=9)
                        ax_s.set_ylabel("% of tokens")
                        ax_s.legend(fontsize=8)
                        ax_s.set_title("Shared tokens — frequency comparison")
                        plt.tight_layout()
                        st.pyplot(fig_s)

                    # ── Distinctive tokens chart ──────────────────────────
                    st.write("**🔀 Distinctive tokens** (unique to each fragrance)")
                    max_rows = max(len(ref_only), len(cand_only), 1)
                    col_r, col_c = st.columns(2)

                    with col_r:
                        st.write(f"*Only in {ref_p[:25]}*")
                        if ref_only:
                            fig_r, ax_r = plt.subplots(
                                figsize=(4, max(2, len(ref_only)*0.35)))
                            ax_r.barh([w for w,_ in ref_only],
                                      [v for _,v in ref_only],
                                      color="#4C9BE8", alpha=0.85)
                            ax_r.invert_yaxis()
                            ax_r.set_xlabel("% of tokens")
                            plt.tight_layout()
                            st.pyplot(fig_r)
                        else:
                            st.info("No exclusive tokens.")

                    with col_c:
                        st.write(f"*Only in {cand[:25]}*")
                        if cand_only:
                            fig_c, ax_c = plt.subplots(
                                figsize=(4, max(2, len(cand_only)*0.35)))
                            ax_c.barh([w for w,_ in cand_only],
                                      [v for _,v in cand_only],
                                      color="#F28C38", alpha=0.85)
                            ax_c.invert_yaxis()
                            ax_c.set_xlabel("% of tokens")
                            plt.tight_layout()
                            st.pyplot(fig_c)
                        else:
                            st.info("No exclusive tokens.")
        else:
            st.info("Select at least one candidate to compare.")


    # ── Tab 3: Factorial Map (Correspondence Analysis) ────────────────────
    with tab3:
        st.subheader("🌐 Correspondence Analysis — Verbatim Token Map")

        # ── Controls ──────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
        with ctrl1:
            ref_products = st.multiselect(
                "📌 Reference Products (benchmark / market)",
                options=p_list,
                default=p_list[:2] if len(p_list) >= 2 else p_list,
                help="Shown as solid circles ●. All others are Candidates ★."
            )
        with ctrl2:
            n_words_show = st.slider(
                "# Token labels to display", min_value=5, max_value=60,
                value=20, step=5,
                help="Top-N most discriminating tokens shown on map."
            )
        with ctrl3:
            show_ellipses = st.toggle("Ellipses", value=True,
                                      help="Draw a rough uncertainty circle around each product.")

        # ── Run CA ────────────────────────────────────────────────────────
        res, err = run_ca(df, p_col, fmin_global)
        if err:
            st.error(err)
        else:
            (row_coords, col_coords, products,
             words, inertia_ratio, word_freqs) = res

            dim1_pct = round(inertia_ratio[0] * 100, 1) if len(inertia_ratio) > 0 else 0
            dim2_pct = round(inertia_ratio[1] * 100, 1) if len(inertia_ratio) > 1 else 0

            # ── Token selection: top-N by CA contribution ─────────────────
            token_dist2   = col_coords[:, 0]**2 + col_coords[:, 1]**2
            token_contrib = word_freqs * token_dist2
            top_idx       = np.argsort(token_contrib)[-n_words_show:]
            show_mask     = np.zeros(len(words), dtype=bool)
            show_mask[top_idx] = True

            # ── Colour palette ─────────────────────────────────────────────
            CAND_COLORS = [
                "#E84393","#FF6B35","#9B59B6","#1ABC9C",
                "#F39C12","#27AE60","#E74C3C","#2980B9",
                "#FF8C00","#00CED1","#8E44AD","#D35400",
            ]
            REF_COLOR = "#2C3E50"

            is_ref = [p in ref_products for p in products]

            # assign a colour to every product
            prod_colors = []
            cand_idx = 0
            for r_flag in is_ref:
                if r_flag:
                    prod_colors.append(REF_COLOR)
                else:
                    prod_colors.append(CAND_COLORS[cand_idx % len(CAND_COLORS)])
                    cand_idx += 1

            # ── Figure ────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(16, 11), facecolor="white")
            ax.set_facecolor("#FAFAFA")

            # Grid + axes
            ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
            ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)
            ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)

            # ── Draw tokens ───────────────────────────────────────────────
            max_freq = word_freqs.max() if word_freqs.max() > 0 else 1
            for i, (w, cx, cy, freq) in enumerate(
                    zip(words, col_coords[:, 0], col_coords[:, 1], word_freqs)):
                if not show_mask[i]:
                    continue
                norm_f = freq / max_freq
                alpha  = 0.35 + norm_f * 0.45       # 0.35–0.80
                fsize  = 8  + norm_f * 10            # 8–18 pt
                ax.plot(cx, cy, marker="x", ms=5,
                        color="#A0522D", alpha=alpha * 0.6, zorder=2)
                txt = ax.text(
                    cx, cy + 0.003, w,
                    fontsize=fsize, ha="center", va="bottom",
                    color="#7B3F00", alpha=alpha, zorder=3,
                    fontweight="bold" if norm_f > 0.6 else "normal",
                )
                txt.set_path_effects([
                    pe.withStroke(linewidth=2.5, foreground="white")
                ])

            # ── Draw products ─────────────────────────────────────────────
            for i, (prod, color, r_flag) in enumerate(
                    zip(products, prod_colors, is_ref)):
                px_  = row_coords[i, 0]
                py_  = row_coords[i, 1]

                if r_flag:
                    marker, ms, lw_edge, zorder = "o", 220, 2.5, 6
                else:
                    marker, ms, lw_edge, zorder = "*", 520, 1.5, 6

                # optional rough ellipse (±1 spread)
                if show_ellipses:
                    n_v  = (df[p_col].astype(str) == str(prod)).sum()
                    # radius shrinks with more respondents (less uncertainty)
                    rad  = max(0.015, 0.06 / (n_v ** 0.3))
                    ell  = Ellipse(
                        (px_, py_), width=rad * 2.8, height=rad * 1.8,
                        angle=0, linewidth=1.2, linestyle="--",
                        edgecolor=color, facecolor=color, alpha=0.10, zorder=4
                    )
                    ax.add_patch(ell)

                ax.scatter(
                    px_, py_,
                    s=ms, marker=marker, color=color,
                    edgecolors="white", linewidths=lw_edge,
                    zorder=zorder, label=f"{'[REF] ' if r_flag else ''}{prod}"
                )

                # Product label — wrapped at 20 chars, with outline
                label_txt = "\n".join(
                    prod[j:j+20] for j in range(0, len(prod), 20)
                )
                txt = ax.text(
                    px_ + 0.004, py_ + 0.006,
                    label_txt,
                    fontsize=9.5,
                    fontweight="bold" if not r_flag else "normal",
                    color=color, zorder=7,
                    va="bottom", ha="left",
                )
                txt.set_path_effects([
                    pe.withStroke(linewidth=3, foreground="white")
                ])

            # ── Quadrant shading (light) ───────────────────────────────────
            x0, x1 = ax.get_xlim() if ax.get_xlim()[0] != 0 else (-0.5, 0.5)
            y0, y1 = ax.get_ylim() if ax.get_ylim()[0] != 0 else (-0.5, 0.5)
            # re-fetch after scatter
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            for qx, qy, clr in [
                (x0, 0,  "#FFF8F0"), (0, 0,   "#F0F8FF"),
                (x0, y0, "#F0FFF0"), (0, y0,  "#FFF0F8"),
            ]:
                qw = (x1 - x0) / 2; qh = (y1 - y0) / 2
                ax.add_patch(patches.Rectangle(
                    (qx, qy), qw, qh,
                    facecolor=clr, alpha=0.25, zorder=0
                ))

            # ── Axes labels & title ───────────────────────────────────────
            ax.set_xlabel(
                f"Dimension 1  —  {dim1_pct}% of inertia",
                fontsize=12, labelpad=8, color="#333333"
            )
            ax.set_ylabel(
                f"Dimension 2  —  {dim2_pct}% of inertia",
                fontsize=12, labelpad=8, color="#333333"
            )
            ax.set_title(
                f"Correspondence Analysis · Token × Product\n"
                f"Total inertia explained: {dim1_pct + dim2_pct:.1f}%",
                fontsize=14, fontweight="bold", pad=14, color="#1a1a1a"
            )

            # ── Legend ────────────────────────────────────────────────────
            handles, labels_ = ax.get_legend_handles_labels()
            legend = ax.legend(
                handles, labels_,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                frameon=True,
                framealpha=0.95,
                edgecolor="#cccccc",
                fontsize=8.5,
                title="Products\n● = Reference   ★ = Candidate",
                title_fontsize=8.5,
            )

            # inertia annotation box (bottom-left, avoids inset_axes unit conflict)
            inertia_txt = (
                f"Dim1: {dim1_pct}%\n"
                f"Dim2: {dim2_pct}%\n"
                f"Total: {dim1_pct + dim2_pct:.1f}%"
            )
            ax.text(
                0.01, 0.01, inertia_txt,
                transform=ax.transAxes,
                fontsize=7.5, va="bottom", ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white", edgecolor="#cccccc",
                    alpha=0.85
                ),
                color="#444444", linespacing=1.5,
            )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Summary metrics ───────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Dim 1 inertia", f"{dim1_pct}%")
            m2.metric("Dim 2 inertia", f"{dim2_pct}%")
            m3.metric("Total explained", f"{dim1_pct + dim2_pct:.1f}%")

            # ── Token proximity table ─────────────────────────────────────
            with st.expander("🔍 Nearest tokens per product (top 10)"):
                prox_rows = []
                for i, prod in enumerate(products):
                    rx, ry = row_coords[i, 0], row_coords[i, 1]
                    dists  = np.sqrt(
                        (col_coords[:, 0] - rx)**2 +
                        (col_coords[:, 1] - ry)**2
                    )
                    for rank, tidx in enumerate(np.argsort(dists)[:10], 1):
                        prox_rows.append({
                            "Product": prod, "Rank": rank,
                            "Token": words[tidx],
                            "Dist":  round(dists[tidx], 4),
                        })
                prox_df = pd.DataFrame(prox_rows)
                pivot   = prox_df.pivot(
                    index="Rank", columns="Product", values="Token")
                st.dataframe(pivot, use_container_width=True)

            # ── Per-product characteristic tokens (bar charts) ────────────
            with st.expander("📊 Characteristic token profiles per product"):
                n_prods = len(products)
                ncols   = min(4, n_prods)
                nrows   = (n_prods + ncols - 1) // ncols
                fig2, axes2 = plt.subplots(
                    nrows, ncols,
                    figsize=(ncols * 4, nrows * 3.5),
                    facecolor="white"
                )
                axes2 = np.array(axes2).flatten()
                for i, (prod, color) in enumerate(zip(products, prod_colors)):
                    ax2 = axes2[i]
                    prod_tok = " ".join(
                        df[df[p_col].astype(str) == str(prod)]["token_str"]
                    ).split()
                    tok_counts = Counter(prod_tok)
                    top8 = tok_counts.most_common(8)
                    if top8:
                        toks_, cnts_ = zip(*top8)
                        ax2.barh(list(toks_), list(cnts_),
                                 color=color, alpha=0.82)
                        ax2.invert_yaxis()
                    short = prod[:22] + "…" if len(prod) > 22 else prod
                    ax2.set_title(short, fontsize=8.5, fontweight="bold",
                                  color=color, pad=4)
                    ax2.tick_params(labelsize=7)
                    ax2.set_xlabel("count", fontsize=7)
                    ax2.set_facecolor("#FAFAFA")
                for j in range(i + 1, len(axes2)):
                    axes2[j].set_visible(False)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

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
                value=", ".join(sorted(st.session_state["ss_stops"])),
                height=200,
                key="stops_textarea"
            )

        with col_right:
            st.write("**Fragrance Merges**")
            merges_text = st.text_area(
                "Edit merges (one per line: variant → canonical)",
                value="\n".join(
                    f"{k} → {v}"
                    for k, v in sorted(st.session_state["ss_merges"].items())
                ),
                height=200,
                key="merges_textarea"
            )

        st.write("**Prefix Strip Protect**")
        protect_text = st.text_area(
            "Edit protected words (comma-separated)",
            value=", ".join(sorted(st.session_state["ss_protect"])),
            height=100,
            key="protect_textarea"
        )

        if st.button("💾 Apply & Re-Process"):
            # Save stopwords to session state
            st.session_state["ss_stops"] = {
                x.strip().lower() for x in stops_text.split(",") if x.strip()
            }
            # Save merges to session state
            new_merges = {}
            for line in merges_text.splitlines():
                if "→" in line:
                    parts = line.split("→", 1)
                    if len(parts) == 2:
                        new_merges[parts[0].strip().lower()] = parts[1].strip().lower()
            st.session_state["ss_merges"] = new_merges
            # Save protect to session state
            st.session_state["ss_protect"] = {
                x.strip().lower() for x in protect_text.split(",") if x.strip()
            }
            # Re-run pipeline on existing data
            if "processed_df" in st.session_state:
                df_reprocess = st.session_state["processed_df"].copy()
                v_col_r = st.session_state["v_col"]
                _ac  = st.session_state["ss_autocorrect"]
                _cat = None if st.session_state["ss_category"] == "None" else st.session_state["ss_category"]
                with st.spinner("Re-processing with new settings…"):
                    df_reprocess["tokens"]    = df_reprocess[v_col_r].apply(
                        lambda x: process_verbatim(x, autocorrect=_ac, category=_cat))
                    df_reprocess["token_str"] = df_reprocess["tokens"].apply(tokens_to_string)
                st.session_state["processed_df"] = df_reprocess
                st.success("✅ Settings updated and data re-processed!")
            else:
                st.success("✅ Settings saved — upload data and run analysis to apply.")

else:
    for tab in [tab1, tab2, tab3, tab4, tab6, tab5]:
        with tab:
            st.info("⬅️ Upload an Excel file and click **🚀 Run Analysis** to get started.")