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
from typing import List, Tuple, Dict
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
# ▌BLOCK 0 — NLTK SETUP
# =============================================================================

@st.cache_resource
def setup_nltk():
    nltk.download("wordnet",                        quiet=True)
    nltk.download("omw-1.4",                        quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

# Optional spaCy FR lemmatizer
@st.cache_resource
def setup_spacy_fr():
    try:
        import spacy
        nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        return nlp
    except Exception:
        return None

_nlp_fr = setup_spacy_fr()

# =============================================================================
# ▌BLOCK 1 — ENGLISH CONSTANTS
# =============================================================================

CONTRACTION_MAP_EN: Dict[str, str] = {
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
_CONTRACTION_RE_EN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP_EN) + r")\b",
    flags=re.IGNORECASE,
)

CLAUSE_SPLITTERS_EN = {"but", "however", "although", "though", "yet", "whereas"}

NEGATIVE_PREFIXES_EN: Tuple[str, ...] = ("un", "in", "im", "ir", "il", "non", "dis")

PREFIX_STRIP_PROTECT_EN = {
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

NEGATION_TERMS_EN = {"not", "never", "no", "without", "nothing", "none", "nobody"}

INTENSITY_MAP_EN: Dict[str, str] = {
    "very": "very", "so": "very", "really": "very", "extremely": "very",
    "incredibly": "very", "super": "very", "highly": "very", "deeply": "very",
    "absolutely": "very", "totally": "very", "quite": "very", "pretty": "very",
    "awfully": "very", "terribly": "very", "remarkably": "very",
}

NEUTRAL_STOPS_EN = {
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
    # ── added ──
    "moment", "moments",
}

FRAGRANCE_MERGES_EN = {
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

EMOTION_BIGRAMS_EN = {("feel", "good"): "feel_good"}

# =============================================================================
# ▌BLOCK 1-FR — FRENCH CONSTANTS
# =============================================================================

CONTRACTION_MAP_FR: Dict[str, str] = {
    "n'est": "ne est", "n'a": "ne a", "n'ai": "ne ai",
    "n'ont": "ne ont", "n'avons": "ne avons", "n'avez": "ne avez",
    "n'était": "ne etait", "n'y": "ne y",
    "j'ai": "j ai", "j'aime": "j aime", "j'adore": "j adore",
    "j'aurais": "j aurais", "j'avais": "j avais", "j'étais": "j etais",
    "c'est": "ce est", "c'était": "ce etait",
    "s'appelle": "se appelle", "s'il": "se il",
    "qu'il": "que il", "qu'elle": "que elle",
    "qu'on": "que on", "qu'un": "que un", "qu'une": "que une",
    "d'une": "de une", "d'un": "de un",
    "d'abord": "de abord", "d'accord": "de accord",
    "m'a": "me a", "m'ont": "me ont",
    "l'odeur": "le odeur", "l'arôme": "le arome",
    "l'air": "le air", "l'eau": "le eau",
}
_FR_ELISION_RE = re.compile(
    r"([a-zàâæçéèêëïîôùûüÿœ]+)['\u2019]([a-zàâæçéèêëïîôùûüÿœ]+)",
    flags=re.IGNORECASE,
)
_CONTRACTION_RE_FR = re.compile(
    r"\b(" + "|".join(
        re.escape(k).replace(r"'", r"['\u2019]") for k in CONTRACTION_MAP_FR
    ) + r")\b",
    flags=re.IGNORECASE,
)

CLAUSE_SPLITTERS_FR = {
    "mais", "cependant", "pourtant", "néanmoins", "toutefois",
    "quoique", "bien que", "quand même", "en revanche", "or",
}

NEGATIVE_PREFIXES_FR: Tuple[str, ...] = (
    "dé", "dés", "més", "mal", "anti", "contre", "non", "in", "im", "ir", "il",
)

PREFIX_STRIP_PROTECT_FR = {
    "intense", "intensité", "intensément", "intoxicant", "intoxicante",
    "incroyable", "incroyablement", "inspirant", "inspirer", "inspiration",
    "intime", "intimité", "indulgent", "indulgence",
    "irrésistible", "irrésistiblement",
    "illuminer", "illuminant", "imaginer", "imagination",
    "immerger", "immersion", "impact", "intéressant",
    "irritant", "irritante", "irritation",
    "majestueux", "majestueuse",
}

NEGATION_TERMS_FR = {
    "pas", "jamais", "plus", "rien", "personne",
    "ni", "guère", "nullement", "aucun", "aucune", "sans", "non",
}

INTENSITY_MAP_FR: Dict[str, str] = {
    "très": "very", "tellement": "very", "vraiment": "very",
    "extrêmement": "very", "absolument": "very", "totalement": "very",
    "trop": "very", "super": "very", "vachement": "very",
    "terriblement": "very", "incroyablement": "very",
    "profondément": "very", "fort": "very", "bien": "very",
    "si": "very", "assez": "very",
    "particulièrement": "very", "notamment": "very", "surtout": "very",
}

NEUTRAL_STOPS_FR = {
    "odeur", "odeurs", "parfum", "parfums", "senteur", "senteurs",
    "arome", "aromes", "note", "notes", "nuance", "nuances",
    "fragrance", "fragrances", "effluve", "effluves",
    "produit", "produits",
    "sentir", "sens", "sensation", "ressenti", "perception",
    "impression", "image", "association", "rappeler", "rappelle",
    "rappelant", "souvenir", "evoquer", "evoque", "sembler", "semble",
    "trouver", "trouve",
    "penser", "pense", "faire", "fait", "laisser", "laisse",
    "aller", "mettre", "met", "prendre", "prend",
    "regarder", "dire", "dit", "utiliser", "convenir",
    "aussi", "bien", "simplement", "quand", "surtout", "donc",
    "comme", "ainsi", "juste", "vraiment", "presque",
    "peu", "beaucoup", "assez", "tres", "plus", "moins",
    "trop", "autant", "tant", "encore", "meme", "quelque",
    "plusieurs", "certain", "certaine", "certains", "certaines",
    "autre", "autres", "propre", "seul", "seule",
    "quelque", "chose", "rien", "tout",
    "je", "j", "me", "moi", "mon", "ma", "mes",
    "nous", "notre", "nos",
    "vous", "votre", "vos",
    "il", "lui", "son", "sa", "ses",
    "elle", "elles", "ils", "leur", "leurs",
    "ce", "se", "on", "y",
    "le", "la", "les", "l",
    "un", "une", "des", "du",
    "est", "suis", "es", "sommes", "etes", "sont",
    "ete", "etre",
    "ai", "as", "avons", "avez", "ont", "avait", "avoir",
    "fait", "fais", "ferai", "ferais",
    "peut", "peux", "pouvons", "pouvez", "peuvent", "pouvoir",
    "vais", "va", "allons", "allez", "vont",
    "doit", "dois", "devons", "devez", "doivent", "devoir",
    "serait", "sera", "serons",
    "de", "du", "des", "d",
    "en", "a", "au", "aux", "par", "pour", "avec", "sur", "sous",
    "dans", "entre", "vers", "depuis", "pendant", "sans", "selon",
    "contre", "avant", "apres", "devant", "derriere",
    "pres", "loin", "autour", "parmi", "sauf",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quoi", "dont",
    "si", "lorsque", "parce", "puisque", "afin",
    "ce", "cet", "cette", "ces", "ceci", "cela", "ca",
    "celui", "celle", "ceux", "celles",
    "comment", "pourquoi", "combien", "quel", "quelle", "quels", "quelles",
    "ne", "pas", "non",
    "tres", "trop", "fort", "bien", "si",
    "alors", "ensuite", "enfin", "deja", "encore",
    "peut-etre", "semble",
    "effectivement", "notamment",
    # ── added ──
    "moment", "moments",
}

FRAGRANCE_MERGES_FR = {
    "fleuri": "fleur", "fleurie": "fleur", "florale": "floral",
    "florales": "floral", "floraux": "floral", "florissant": "fleur",
    "epanoui": "fleur", "epanouie": "fleur",
    "fraicheur": "frais", "fraiche": "frais", "rafraichissant": "frais",
    "propre": "clean", "proprete": "clean", "nettete": "clean",
    "relaxant": "relax", "relaxante": "relax",
    "reposant": "relax", "reposante": "relax",
    "detendre": "relax", "detendu": "relax", "detendue": "relax",
    "relaxation": "relax",
    "confort": "comfort", "confortable": "comfort",
    "reconfortant": "comfort", "reconfortante": "comfort",
    "douillet": "comfort", "douillette": "comfort",
    "boise": "bois", "boisee": "bois",
    "ligneuse": "bois", "ligneux": "bois",
    "fruite": "fruit", "fruitee": "fruit",
    "epice": "epice", "epicee": "epice",
    "musque": "musc", "musquee": "musc",
    "sucre": "sucre", "sucree": "sucre",
    "douceur": "doux", "douce": "doux",
    "marin": "ocean", "marine": "ocean",
    "aquatique": "aquatic", "aqueux": "aquatic", "aqueuse": "aquatic",
    "oceanique": "ocean", "mer": "ocean",
    "poudreux": "poudre", "poudreuse": "poudre", "poudree": "poudre",
    "fume": "fumee",
    "herbace": "vert", "herbacee": "vert",
    "herbeux": "vert", "herbeuse": "vert",
    "vegetal": "vert", "vegetale": "vert",
    "gazon": "vert", "feuille": "vert", "feuilles": "vert",
    "citronne": "citrus", "citrus": "citrus",
    "agrume": "citrus", "agrumes": "citrus",
    "vieillot": "old_fashioned", "demode": "old_fashioned",
    "mamie": "grandmere", "papi": "grandpere",
    "meme": "grandmere", "pepe": "grandpere",
}

EMOTION_BIGRAMS_FR = {
    ("sent",  "bon"):   "good_smell",
    ("bonne", "odeur"): "good_smell",
    ("se",    "sentir"):"feel",
    ("bien",  "etre"):  "wellbeing",
    ("coup",  "coeur"): "coup_de_coeur",
}

# =============================================================================
# ▌BLOCK 1-CAT — CATEGORY-SPECIFIC STOPWORDS (EN + FR)
# =============================================================================

CATEGORY_STOPS: Dict[str, Dict[str, set]] = {
    "fabric_care": {
        "en": {
            "laundry","wash","washing","washed","clothes","clothing","cloth",
            "garment","garments","fabric","fabrics","textile","textiles",
            "load","machine","drum","cycle","rinse","spin","dry","drying",
            "dried","tumble","dryer","detergent","powder","liquid","capsule",
            "pod","tablet","dose","dosage","stain","stains","dirt","dirty",
            "soil","soiled","clean","cleaning","cleaner","cleanse",
            "whitening","brightening","colour","color","white","bright",
        },
        "fr": {
            "lessive","lavage","laver","lave","relavage","vetement","vetements",
            "habits","tissu","tissus","textile","textiles","linge","machine",
            "tambour","cycle","rincage","essorage","sechage","secher","seche",
            "detergent","poudre","liquide","capsule","dosette","dose",
            "tache","taches","sale","salissure","souillure",
            "propre","nettoyer","nettoyage","blancheur","blanchiment","blanc",
        },
    },
    "fabric_softener": {
        "en": {
            "softener","conditioner","softening","soften","laundry","wash",
            "washing","washed","clothes","clothing","fabric","fabrics","linen",
            "dry","drying","tumble","dryer","rinse","cycle","machine",
            "static","wrinkle","crease","fluffy","fluffiness","soft","softness",
        },
        "fr": {
            "adoucissant","assouplissant","adoucir","assouplir","lessive",
            "lavage","laver","vetement","vetements","linge","tissu","tissus",
            "sechage","secher","rincage","cycle","machine",
            "statique","froise","moelleux","doux","douceur",
        },
    },
    "dishwashing": {
        "en": {
            "dish","dishes","dishwash","dishwasher","dishwashing","plate",
            "plates","bowl","bowls","glass","glasses","cutlery","utensil",
            "utensils","pan","pans","pot","pots","grease","greasy","fat",
            "residue","rinse","tablet","pod","capsule","foam","foaming",
            "lather","bubble","bubbles","wash","washing","washed",
            "clean","cleaning","cleaner","streak","spot","shine",
        },
        "fr": {
            "vaisselle","laver","lavage","lave-vaisselle","assiette","assiettes",
            "verre","verres","bol","couverts","ustensile","ustensiles",
            "poele","casserole","graisse","gras","residu","rincage","pastille",
            "capsule","dosette","mousse","moussant","bulle","bulles",
            "nettoyer","nettoyage","trace","brillance",
        },
    },
    "surface_cleaner": {
        "en": {
            "surface","surfaces","counter","countertop","worktop","floor",
            "tile","tiles","bathroom","kitchen","toilet","wipe","wiping",
            "wiped","spray","spraying","scrub","scrubbing","rinse","bucket",
            "bacteria","germ","germs","disinfect","disinfectant","mould",
            "mold","limescale","grease","clean","cleaning","cleaner",
            "cleanse","bleach","chlorine",
        },
        "fr": {
            "surface","surfaces","plan de travail","comptoir","sol","carrelage",
            "salle de bain","cuisine","toilette","essuyer","essuyage",
            "pulveriser","spray","frotter","rincage","seau","bacterie",
            "germe","germes","desinfecter","desinfectant","moisissure",
            "tartre","graisse","nettoyer","nettoyage","nettoyant","javel","chlore",
        },
    },
    "fine_fragrance": {
        "en": {
            "perfume","fragrance","scent","spray","bottle","eau","parfum",
            "toilette","cologne","apply","applied","spritz","dab","wrist",
            "neck","wear","wearing","wore","last","lasts","lasting",
            "longevity","sillage","trail","top","middle","base","accord",
            "blend","nose","house","brand","designer","niche",
        },
        "fr": {
            "parfum","fragrance","senteur","vaporisateur","flacon",
            "eau de parfum","eau de toilette","cologne","appliquer",
            "vaporiser","poignet","cou","porter","porte","tenue","sillage",
            "traine","tete","coeur","fond","accord","nez","maison","marque",
        },
    },
    "body_care": {
        "en": {
            "shower","bath","bathing","body","skin","lather","rinse","wash",
            "washing","gel","lotion","cream","moisturise","moisturize",
            "moisturiser","moisturizer","absorb","absorption","texture",
            "consistency","apply","applied","rub","massage","dry","dryness",
            "oily","sensitive",
        },
        "fr": {
            "douche","bain","corps","peau","mousse","rincage","laver","gel",
            "lotion","creme","hydrater","hydratant","hydratation","absorber",
            "absorption","texture","consistance","appliquer","frictionner",
            "masser","seche","secheresse","gras","sensible",
        },
    },
    "hair_care": {
        "en": {
            "hair","shampoo","conditioner","scalp","strand","strands","wash",
            "washing","rinse","lather","foam","dry","drying","blow","volume",
            "frizz","damage","damaged","repair","strengthen","strength",
            "colour","color","dye","bleach","greasy","oily","dandruff",
            "itchy","smooth","smoothness","silky","shiny","shine",
        },
        "fr": {
            "cheveux","shampoing","apres-shampoing","cuir chevelu","meche",
            "meches","laver","lavage","rincage","mousse","sechage","secher",
            "volume","frisottis","abime","reparation","renforcer","couleur",
            "teinture","decoloration","gras","pellicules","demangeaison",
            "lisse","soyeux","brillance",
        },
    },
    "deodorant": {
        "en": {
            "deodorant","deo","antiperspirant","spray","roll","underarm",
            "armpit","sweat","sweating","perspiration","protect","protection",
            "hour","hours","day","apply","applied","dry","skin","irritation",
            "sensitive","body odour","body odor","odour","odor",
        },
        "fr": {
            "deodorant","deo","antitranspirant","spray","bille","aisselle",
            "transpiration","sudation","proteger","protection","heure",
            "heures","journee","appliquer","secher","peau","irritation",
            "sensible","odeur corporelle",
        },
    },
    "oral_care": {
        "en": {
            "tooth","teeth","toothpaste","toothbrush","mouthwash","brush",
            "brushing","rinse","spit","cavity","plaque","tartar","whitening",
            "enamel","gum","gums","breath","clean","cleaning","foam","foaming",
            "mint","minty",
        },
        "fr": {
            "dent","dents","dentifrice","brosse a dents","bain de bouche",
            "brosser","brossage","rincage","cracher","carie","plaque",
            "tartre","blanchiment","email","gencive","gencives","haleine",
            "nettoyer","mousse","menthe",
        },
    },
    "air_care": {
        "en": {
            "air","room","space","home","house","office","spray","diffuser",
            "candle","plug","plugin","wick","burn","burning","melt","wax",
            "neutralise","neutralize","eliminate","mask","refresh",
            "refreshing","freshen","last","lasts","lasting","hours","hour",
        },
        "fr": {
            "air","piece","maison","bureau","espace","spray","diffuseur",
            "bougie","prise","meche","bruler","fondre","cire","neutraliser",
            "eliminer","masquer","rafraichir","rafraichissant","duree",
            "tenir","heure","heures",
        },
    },
    "baby_care": {
        "en": {
            "baby","infant","newborn","child","toddler","nappy","diaper",
            "wipe","wipes","bath","bathing","wash","lotion","cream","powder",
            "gentle","mild","hypoallergenic","skin","rash",
        },
        "fr": {
            "bebe","nourrisson","nouveau-ne","enfant","bambin","couche",
            "lingette","lingettes","bain","laver","lotion","creme","poudre",
            "doux","douceur","hypoallergenique","peau","erytheme",
        },
    },
    "skincare": {
        "en": {
            "skin","face","facial","serum","cream","moisturiser","moisturizer",
            "lotion","gel","toner","essence","apply","applied","absorb",
            "absorption","texture","consistency","layer","routine","dry",
            "dryness","oily","combination","sensitive","pore","pores",
            "wrinkle","wrinkles","aging","ageing","spf","sunscreen","sunblock",
            "hydrate","hydrating","hydration",
        },
        "fr": {
            "peau","visage","facial","serum","creme","hydratant","lotion",
            "gel","tonique","essence","appliquer","absorber","absorption",
            "texture","consistance","couche","routine","seche","secheresse",
            "gras","mixte","sensible","pore","pores","ride","rides",
            "vieillissement","spf","ecran solaire","hydrater","hydratation",
        },
    },
}

# Human-readable labels for the UI selectbox
CATEGORY_DISPLAY: Dict[str, str] = {
    "none":            "— None (universal stops only) —",
    "fabric_care":     "🧺 Fabric Care (laundry detergent)",
    "fabric_softener": "🌸 Fabric Softener / Conditioner",
    "dishwashing":     "🍽️ Dishwashing",
    "surface_cleaner": "🧹 Surface Cleaner",
    "fine_fragrance":  "🌹 Fine Fragrance (EdP / EdT)",
    "body_care":       "🚿 Body Care (shower gel / lotion)",
    "hair_care":       "💇 Hair Care (shampoo / conditioner)",
    "deodorant":       "💨 Deodorant / Antiperspirant",
    "oral_care":       "🦷 Oral Care (toothpaste / mouthwash)",
    "air_care":        "🕯️ Air Care (freshener / candle)",
    "baby_care":       "🍼 Baby Care",
    "skincare":        "✨ Skincare (face cream / serum)",
}

def get_category_stops(category_key: str, lang: str) -> set:
    if not category_key or category_key == "none":
        return set()
    return CATEGORY_STOPS.get(category_key, {}).get(lang, set())

# =============================================================================
# ▌BLOCK 2 — SESSION STATE INIT
# =============================================================================

if "ss_stops_en" not in st.session_state:
    st.session_state["ss_stops_en"] = set(NEUTRAL_STOPS_EN)
if "ss_stops_fr" not in st.session_state:
    st.session_state["ss_stops_fr"] = set(NEUTRAL_STOPS_FR)
if "ss_merges" not in st.session_state:
    st.session_state["ss_merges"] = dict(FRAGRANCE_MERGES_EN)
if "ss_protect" not in st.session_state:
    st.session_state["ss_protect"] = set(PREFIX_STRIP_PROTECT_EN)
if "ss_lang" not in st.session_state:
    st.session_state["ss_lang"] = "en"
if "ss_category" not in st.session_state:
    st.session_state["ss_category"] = "none"

# =============================================================================
# ▌BLOCK 3 — PIPELINE STEPS (bilingual)
# =============================================================================

_PUNCT_STRIP = re.compile(r"[" + re.escape(string.punctuation.replace("-", "").replace("'", "")) + r"]")

def _expand_fr_elision_generic(text: str) -> str:
    return _FR_ELISION_RE.sub(r"\1 \2", text)

def step1_char_normalize(text: str, lang: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u00ab", "").replace("\u00bb", "")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if lang == "fr":
        for src, dst in [("œ","oe"),("æ","ae"),("à","a"),("â","a"),
                         ("é","e"),("è","e"),("ê","e"),("ë","e"),
                         ("î","i"),("ï","i"),("ô","o"),
                         ("ù","u"),("û","u"),("ü","u"),("ÿ","y"),("ç","c")]:
            text = text.replace(src, dst)
        text = re.sub(r"-", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def step2_expand_contractions(text: str, lang: str) -> str:
    if lang == "fr":
        def _fr_replace(m):
            key = m.group(0).lower().replace("\u2019", "'")
            return CONTRACTION_MAP_FR.get(key, m.group(0))
        text = _CONTRACTION_RE_FR.sub(_fr_replace, text)
        text = _expand_fr_elision_generic(text)
        return text
    return _CONTRACTION_RE_EN.sub(lambda m: CONTRACTION_MAP_EN[m.group(0).lower()], text)

def step3_clause_segment(text: str, lang: str) -> List[str]:
    splitters = CLAUSE_SPLITTERS_FR if lang == "fr" else CLAUSE_SPLITTERS_EN
    single = sorted([s for s in splitters if " " not in s], key=len, reverse=True)
    multi  = sorted([s for s in splitters if " " in s],     key=len, reverse=True)
    pattern = r"(?:,\s*|\s+)(?:" + "|".join(re.escape(s) for s in (multi + single)) + r")(?:\s+|,\s*)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

def step4_tokenize(text: str) -> List[str]:
    return [t for t in _PUNCT_STRIP.sub(" ", text).split() if t]

def step5_prefix_strip(tokens: List[str], lang: str) -> List[str]:
    protect = st.session_state["ss_protect"]
    prefixes = NEGATIVE_PREFIXES_FR if lang == "fr" else NEGATIVE_PREFIXES_EN
    result = []
    for tok in tokens:
        if tok in protect:
            result.append(tok); continue
        stripped = False
        for pref in prefixes:
            if tok.startswith(pref) and len(tok) > len(pref) + 2:
                root = tok[len(pref):]
                if lang == "fr" and len(root) < 3:
                    break
                result.extend(["not", root]); stripped = True; break
        if not stripped: result.append(tok)
    return result

def _get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "V": wordnet.VERB,
            "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def step6_lemmatize(tokens: List[str], lang: str) -> List[str]:
    if lang == "fr":
        fr_keep = NEGATION_TERMS_FR | set(INTENSITY_MAP_FR.keys())
        if _nlp_fr is not None:
            out = []
            for tok in tokens:
                if tok in fr_keep or "_" in tok:
                    out.append(tok)
                else:
                    doc = _nlp_fr(tok)
                    out.append(doc[0].lemma_ if doc else tok)
            return out
        return tokens   # no-op fallback
    # English
    out = []
    for tok in tokens:
        if tok in NEGATION_TERMS_EN or tok in INTENSITY_MAP_EN:
            out.append(tok)
        else:
            out.append(lemmatizer.lemmatize(tok, pos=_get_wordnet_pos(tok)))
    return out

def step5_5_synonym_merge(tokens: List[str], lang: str) -> List[str]:
    merges = st.session_state["ss_merges"]
    base   = FRAGRANCE_MERGES_FR if lang == "fr" else FRAGRANCE_MERGES_EN
    merged = {**base, **merges}   # user edits override defaults
    return [merged.get(tok, tok) for tok in tokens]

def step_fr_negation_collapse(tokens: List[str]) -> List[str]:
    FR_NEG_PARTICLES = {"pas", "jamais", "plus", "rien", "guere", "nullement"}
    out = []; i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "ne" and i + 2 < len(tokens) and tokens[i+2] in FR_NEG_PARTICLES:
            target = tokens[i+1]
            if "_" not in target and target not in FR_NEG_PARTICLES:
                out.append(f"not_{target}"); i += 3; continue
        if tok == "ne" and i + 1 < len(tokens) and tokens[i+1] in FR_NEG_PARTICLES:
            out.append("not"); i += 2; continue
        if tok == "pas" and i + 1 < len(tokens):
            nxt = tokens[i+1]
            if "_" not in nxt and nxt not in FR_NEG_PARTICLES and nxt != "ne":
                out.append(f"not_{nxt}"); i += 2; continue
        out.append(tok); i += 1
    return out

def step_emotion_bigrams(tokens: List[str], lang: str) -> List[str]:
    bigrams = EMOTION_BIGRAMS_FR if lang == "fr" else EMOTION_BIGRAMS_EN
    out = []; i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            pair = (tokens[i], tokens[i+1])
            if pair in bigrams:
                out.append(bigrams[pair]); i += 2; continue
        out.append(tokens[i]); i += 1
    return out

def step7_stopword_removal(tokens: List[str], lang: str, cat_stops: set) -> List[str]:
    if lang == "fr":
        stops  = st.session_state["ss_stops_fr"]
        neg    = NEGATION_TERMS_FR
        intens = INTENSITY_MAP_FR
    else:
        stops  = st.session_state["ss_stops_en"]
        neg    = NEGATION_TERMS_EN
        intens = INTENSITY_MAP_EN
    effective = stops | cat_stops
    return [t for t in tokens if t in neg or t in intens or "_" in t or t not in effective]

def step7_5_intensity_normalize(tokens: List[str], lang: str) -> List[str]:
    imap = INTENSITY_MAP_FR if lang == "fr" else INTENSITY_MAP_EN
    return [imap.get(t, t) for t in tokens]

def step8_ngram_collapse(tokens: List[str], lang: str) -> List[str]:
    neg_all = NEGATION_TERMS_FR | NEGATION_TERMS_EN
    out = []; i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "_" in tok:
            out.append(tok); i += 1; continue
        if tok == "not" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt == "very" and i + 2 < len(tokens):
                tgt = tokens[i + 2]
                if "_" not in tgt and tgt not in neg_all:
                    out.append(f"not_{tgt}"); i += 3; continue
            if nxt not in neg_all and "_" not in nxt:
                out.append(f"not_{nxt}"); i += 2; continue
        if tok == "very" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in neg_all and nxt != "very" and "_" not in nxt:
                out.append(f"very_{nxt}"); i += 2; continue
        out.append(tok); i += 1
    return out

def step9_dedup(tokens: List[str]) -> List[str]:
    seen = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]

def process_verbatim(raw_text: str) -> List[str]:
    if not raw_text or not isinstance(raw_text, str): return []
    lang     = st.session_state["ss_lang"]
    cat_key  = st.session_state["ss_category"]
    cat_stops = get_category_stops(cat_key, lang)

    text = step1_char_normalize(raw_text, lang)
    text = step2_expand_contractions(text, lang)
    all_tokens = []
    for clause in step3_clause_segment(text, lang):
        tokens = step4_tokenize(clause)
        if lang == "fr":
            tokens = step_fr_negation_collapse(tokens)
        tokens = step5_prefix_strip(tokens, lang)
        tokens = step6_lemmatize(tokens, lang)
        tokens = step5_5_synonym_merge(tokens, lang)
        tokens = step_emotion_bigrams(tokens, lang)
        tokens = step7_stopword_removal(tokens, lang, cat_stops)
        tokens = step7_5_intensity_normalize(tokens, lang)
        tokens = step8_ngram_collapse(tokens, lang)
        all_tokens.extend(tokens)
    return step9_dedup(all_tokens)

def tokens_to_string(tokens: List[str]) -> str:
    return " ".join(tokens)

def get_weight(token: str) -> float:
    return 1.5 if token.startswith("very_") else 1.0

def get_bucket(token: str) -> str:
    if token.startswith("not_"):  return "negative"
    if token.startswith("very_"): return "positive_amplified"
    return "positive"


# =============================================================================
# ▌BLOCK 4 — VISUALISATION HELPERS
# =============================================================================

def generate_word_cloud(token_series: pd.Series, palette: str, shape: str):
    weight_counter: Counter = Counter()
    for tokens in token_series:
        if isinstance(tokens, list):
            for t in tokens:
                weight_counter[t.replace("_", " ")] += get_weight(t)
    if not weight_counter:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha="center"); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20, 20, 780, 780), fill=0)
        mask = np.array(img)
    wc = WordCloud(
        background_color="white", colormap=palette, mask=mask,
        width=800, height=500, collocations=False, regexp=r"\S+"
    ).generate_from_frequencies(weight_counter)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    return fig


def generate_word_tree_advanced(token_series: pd.Series, min_freq: int, palette: str):
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
                    ax.add_patch(patches.Polygon(pts[hull.vertices], closed=True,
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
# ▌BLOCK 5 — SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # ── Language selector ─────────────────────────────────────────────────────
    st.markdown("**🌐 Language**")
    lang_choice = st.radio(
        "Verbatim language",
        options=["🇬🇧 English", "🇫🇷 French"],
        index=0 if st.session_state["ss_lang"] == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_lang = "fr" if "French" in lang_choice else "en"
    if new_lang != st.session_state["ss_lang"]:
        st.session_state["ss_lang"] = new_lang
        # reset merges/protect to match new language defaults
        if new_lang == "fr":
            st.session_state["ss_merges"]  = dict(FRAGRANCE_MERGES_FR)
            st.session_state["ss_protect"] = set(PREFIX_STRIP_PROTECT_FR)
        else:
            st.session_state["ss_merges"]  = dict(FRAGRANCE_MERGES_EN)
            st.session_state["ss_protect"] = set(PREFIX_STRIP_PROTECT_EN)

    lang_label = "🇫🇷 French" if st.session_state["ss_lang"] == "fr" else "🇬🇧 English"
    st.caption(f"Active: **{lang_label}**")

    st.divider()

    # ── Category selector ─────────────────────────────────────────────────────
    st.markdown("**🗂️ Product Category**")
    cat_options    = list(CATEGORY_DISPLAY.keys())
    cat_labels     = list(CATEGORY_DISPLAY.values())
    current_cat    = st.session_state["ss_category"]
    current_idx    = cat_options.index(current_cat) if current_cat in cat_options else 0
    selected_label = st.selectbox(
        "Category stopwords",
        options=cat_labels,
        index=current_idx,
        label_visibility="collapsed",
    )
    selected_cat = cat_options[cat_labels.index(selected_label)]
    if selected_cat != st.session_state["ss_category"]:
        st.session_state["ss_category"] = selected_cat

    if selected_cat != "none":
        preview_stops = get_category_stops(selected_cat, st.session_state["ss_lang"])
        st.caption(f"**{len(preview_stops)}** domain words will be filtered")
        with st.expander("Preview category stops", expanded=False):
            st.write(", ".join(sorted(preview_stops)) or "—")

    st.divider()

    # ── File upload & column selection ────────────────────────────────────────
    uploaded_file = st.file_uploader("📂 Upload Excel", type=["xlsx"])

    fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
    use_tfidf   = st.toggle("Use TF-IDF Weighting", value=True)
    shape_opt   = st.radio("Cloud Shape", ["Rectangle", "Round"])
    palette_opt = st.selectbox("Palette", ["copper","GnBu","RdPu","viridis","Spectral"])

    if uploaded_file:
        try:
            xl     = pd.ExcelFile(uploaded_file)
            sheet  = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)

            filter_col   = st.selectbox("Filter Column:", ["No Filter"] + list(df_raw.columns))
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
        _lang    = st.session_state["ss_lang"]
        _cat     = st.session_state["ss_category"]
        _cat_lbl = CATEGORY_DISPLAY.get(_cat, _cat)

        with st.spinner(f"Running pipeline [{lang_label} · {_cat_lbl}]…"):
            df_filtered["tokens"]    = df_filtered[v_col].apply(process_verbatim)
            df_filtered["token_str"] = df_filtered["tokens"].apply(tokens_to_string)

        st.session_state["processed_df"] = df_filtered
        st.session_state["filter_info"]  = filter_label
        st.session_state["pref_col"]     = s_col
        st.session_state["p_col"]        = p_col
        st.session_state["v_col"]        = v_col
        st.success(f"✅ Processed {len(df_filtered)} verbatims  [{lang_label} · {_cat_lbl}]")


# =============================================================================
# ▌BLOCK 6 — TABS
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

    # ── Tab 1: Single Product ─────────────────────────────────────────────────
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
        tree_fig = generate_word_tree_advanced(product_data["tokens"], fmin_global, palette_opt)
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

    # ── Tab 2: Comparison ─────────────────────────────────────────────────────
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

    # ── Tab 3: Factorial Map ──────────────────────────────────────────────────
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

    # ── Tab 4: Topic Lab ──────────────────────────────────────────────────────
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

    # ── Tab 5 (tab6): Impact Lab ──────────────────────────────────────────────
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

    # ── Tab 6 (tab5): Exclusions & Grams ─────────────────────────────────────
    with tab5:
        st.subheader("🚫 Exclusions & Synonym Lab")

        active_lang = st.session_state["ss_lang"]
        stops_key   = "ss_stops_fr" if active_lang == "fr" else "ss_stops_en"

        col_left, col_right = st.columns(2)

        with col_left:
            lang_flag = "🇫🇷" if active_lang == "fr" else "🇬🇧"
            st.write(f"**{lang_flag} Stopwords (active language)**")
            stops_text = st.text_area(
                "Edit stopwords (comma-separated)",
                value=", ".join(sorted(st.session_state[stops_key])),
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
            st.session_state[stops_key] = {
                x.strip().lower() for x in stops_text.split(",") if x.strip()
            }
            new_merges = {}
            for line in merges_text.splitlines():
                if "→" in line:
                    parts = line.split("→", 1)
                    if len(parts) == 2:
                        new_merges[parts[0].strip().lower()] = parts[1].strip().lower()
            st.session_state["ss_merges"]  = new_merges
            st.session_state["ss_protect"] = {
                x.strip().lower() for x in protect_text.split(",") if x.strip()
            }
            if "processed_df" in st.session_state:
                df_reprocess = st.session_state["processed_df"].copy()
                v_col_r = st.session_state["v_col"]
                with st.spinner("Re-processing with new settings…"):
                    df_reprocess["tokens"]    = df_reprocess[v_col_r].apply(process_verbatim)
                    df_reprocess["token_str"] = df_reprocess["tokens"].apply(tokens_to_string)
                st.session_state["processed_df"] = df_reprocess
                st.success("✅ Settings updated and data re-processed!")
            else:
                st.success("✅ Settings saved — upload data and run analysis to apply.")

else:
    for tab in [tab1, tab2, tab3, tab4, tab6, tab5]:
        with tab:
            st.info("⬅️ Upload an Excel file and click **🚀 Run Analysis** to get started.")