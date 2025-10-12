import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer

def _ensure_nltk():
    try:
        _ = stopwords.words("german")
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()

GERMAN_SW = set(stopwords.words("german")) | {"stuttgart", "stgt", "https", "http"}
ENGLISH_SW = set(stopwords.words("english"))

# Zusätzliche Füll-/Funktionswörter, die i. d. R. nicht themenbildend sind
FILLER_DE = {
    "heute","jemand","eigentlich","wirklich","direkt","danke","bitte","gerne",
    "halt","ne","ach","wow","super","bisschen","ziemlich","total","richtig",
    "viele","vielleicht","wurde","werden","wird","waren","sein","wäre","hat",
    "hast","habt","paar","seite","alleine","menschen","leute","deutschland",
    "stadt","ganz","schön","warum","wochen","neu","neue","minuten","kommt",
    "kommen","geht","gehen","gesehen","sieht","richtig","bereits","jedoch",
    "sowie","außerdem","daher","deshalb","trotzdem","überhaupt","dabei",
    "danach","vorher","zwei","drei","vier","fünf","jahren","jahr",
        "sowas","meinung","leben","etc","los","weniger","glück","unterwegs","damals",
    "besonders","verstehe","ersten","wahrscheinlich","bestimmt","vermutlich",
    "endlich","vorbei","genug","nein","bild","finden","steht","art","nen","kurz",
    "oben","liegt","lassen","lange","zurück","komplett",
    # Subreddit-/Region-spezifisch
    "stuttgarter","stuttgart","stgt","deutschland",
    # Generierte Liste aus Ordner Optional
    "ab","aber","abseits","abzüglich","ach","achja","achnee","acht","achte","achter","achtzehn","achtzig","achtzigste","achtzigster","ah","aha","alle","allein","allem","allen","aller","allerdings","alles","allesamt","als","also","am","an","andere","andererseits","anfangs","angesichts","anhand","anlässlich","ans","anschließend","ansonst","ansonsten","anstatt","anstelle","aua","auch","auf","aufgrund","aus","ausgenommen","ausschliesslich","ausschließlich","ausser","ausserhalb","autsch","außer","außerhalb","bei","beide","beiden","beim","bevor","beziehungsweise","bezüglich","bin","binnen","bis","bist","bleibe","bleiben","bleibest","bleibet","bleibst","bleibt","blieb","bliebe","blieben","bliebest","bliebet","bliebst","bliebt","bloss","bloß","boah","bravo","brrr","bäh","da","dabei","dadurch","dafür","dagegen","damit","danach","dann","darauf","darf","darfst","darum","das","dasjenige","dass","dasselbe","dazu","dein","deine","deiner","deines","deins","dem","demjenigen","demselben","den","denen","denjenigen","denn","denselben","der","deren","derjenige","derjenigen","derselbe","derselben","des","deshalb","desjenigen","desselben","dessen","desto","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","diesseits","dir","doch","drei","dreizehn","dreißig","dreißigste","dreißigster","dritte","dritter","du","durch","durfte","durften","durftest","durftet","dürfe","dürfen","dürfest","dürfet","dürft","dürfte","dürften","dürftest","dürftet","ebenso","ehe","ei","ein","eine","einem","einen","einer","einerseits","eines","einige","einigen","einiger","eins","einschliesslich","einschließlich","elf","entgegen","entlang","entsprechend","entweder","er","erste","erster","es","etliche","etlichen","etliches","etwas","euch","euer","euere","euerem","eueren","euers","eure","eurem","euren","eures","exklusive","ey","falls","fern","fernab","ferner","folglich","fünf","fünfte","fünfter","fünfzehn","fünfzig","fünfzigste","fünfzigster","für","geblieben","gedurft","gegen","gehabt","geil","gekonnt","gemocht","gemusst","gemußt","gemäss","gemäß","genau","genauso","geschweige","gesollt","getreu","gewesen","gewollt","geworden","gleich","gleichwie","habe","haben","habest","habet","habt","hach","haha","halber","hallo","hast","hat","hatte","hatten","hattest","hattet","he","heda","hehe","hey","hinsichtlich","hinter","hm","hoho","holla","hopp","hoppla","huch","hui","hundert","hundertste","hundertster","hurra","husch","hä","hätte","hätten","hättest","hättet","höh","ich","igitt","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","ihrs","im","immerhin","in","indem","indes","indessen","infolge","inklusive","inmitten","innerhalb","ins","insofern","inzwischen","irgendein","irgendeine","irgendeinem","irgendeinen","irgendeines","irgendeins","irgendetwas","irgendjemand","irgendjemandem","irgendjemanden","irgendwelche","irgendwelchem","irgendwelchen","irgendwelcher","irgendwelches","ist","je","jeder","jedoch","jene","jenem","jenen","jener","jenes","jenseits","jesses","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","keines","keins","konnte","konnten","konntest","konntet","kontra","kraft","könne","können","könnest","könnet","könnt","könnte","könnten","könntest","könntet","lang","laut","links","lol","längs","mag","magst","man","manch","manche","manchem","manchen","mancher","manches","mangels","mehrere","mehrerer","mein","meine","meiner","meines","meins","mhm","mich","milliarden","millionen","mir","mit","mitsamt","mittels","mmmh","mochte","mochten","mochtest","mochtet","muss","musst","musste","mussten","musstet","muß","mußt","mußte","mußten","mußtet","möchte","möchten","möchtest","möchtet","möge","mögen","mögest","möget","mögt","müsse","müssen","müssest","müsset","müsst","müße","müßest","müßet","müßt","müßte","müßtest","müßtet","nach","nachdem","nah","nahe","naja","ne","neben","nebst","neun","neunte","neunter","neunzehn","neunzig","neunzigste","neunzigster","nichts","niemand","niemandem","niemanden","noch","nun","nur dass","nächst","nämlich","nördlich","ob","oberhalb","obgleich","obschon","obwohl","obzwar","oder","oh","ohje","ohne","oho","olala","pah","peng","per","pfui","pro","psst","puh","rechts","samt","schliesslich","schließlich","schwupp","sechs","sechste","sechster","sechzehn","sechzig","sechzigste","sechzigster","sei","seid","seien","seiet","sein","seine","seiner","seines","seins","seist","seit","seitdem","seitens","seitlich","sich","sie","sieben","siebte","siebter","siebzehn","siebzig","siebzigste","siebzigster","sind","so","sobald","sodass","sofern","solange","solche","solchem","solchen","solcher","solches","soll","solle","sollen","sollest","sollet","sollst","sollt","sollte","sollten","solltest","solltet","sondern","sonst","sooft","soviel","soweit","sowenig","sowie","sowohl","statt","sämtlich","südlich","tada","tausend","tausendste","tausendster","tja","trara","trotz","trotzdem","ts","uff","um","umso","und","ungeachtet","uns","unser","unsere","unseres","unsers","unter","unterhalb","unweit","ups","viel","viele","vielem","vielen","vieler","vieles","vier","vierte","vierter","vierzehn","vierzig","vierzigste","vierzigster","vom","von","vor","vorbehaltlich","vorher","wann","war","waren","warst","wart","was","weder","wegen","weil","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","wenigen","weniges","wenn","wenngleich","wennschon","wer","werde","werden","werdet","wessen","westlich","wie","wieweit","will","willst","wir","wird","wirst","wo","wofern","wohingegen","wolle","wollen","wollest","wollet","wollt","wollte","wollten","wolltest","wolltet","wurde","wurden","wurdest","wurdet","während","währenddem","währenddessen","wäre","wären","wärest","wäret","wärst","wärt","würde","würden","würdest","würdet","zehn","zehnte","zehnter","zu","zudem","zugunsten","zum","zumal","zur","zusätzlich","zuvor","zwanzig","zwanzigste","zwanzigster","zwar","zwecks","zwei","zweiter","zwischen","zwölf","östlich","über"
}
GERMAN_SW |= FILLER_DE

URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\wäöüÄÖÜß\s]")

STEMMER = GermanStemmer(ignore_stopwords=True)

def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(text: str):
    return re.findall(r"\b[\wäöüÄÖÜß]{2,}\b", text)

def detect_language_by_stopword_ratio(tokens, min_tokens=12, margin=1.15):
    """Heuristik: vergleicht Stopwort-Anteile Deutsch vs. Englisch."""
    if not tokens:
        return "unknown"
    n = len(tokens)
    if n < min_tokens:
        return "unknown"
    de_hits = sum(1 for t in tokens if t in GERMAN_SW)
    en_hits = sum(1 for t in tokens if t in ENGLISH_SW)
    de_ratio = de_hits / n
    en_ratio = en_hits / n
    if de_ratio >= en_ratio * margin and de_ratio > 0:
        return "de"
    if en_ratio >= de_ratio * margin and en_ratio > 0:
        return "en"
    return "unknown"

def add_clean_columns(df: pd.DataFrame, use_stemming: bool=False,
                      filter_english: bool=True, include_comments: bool=False) -> pd.DataFrame:
    """
    Erweitert df um:
      - text_all, clean, tokens_raw, lang, tokens, clean_for_vect
    Achtung: Standard ohne Stemming -> bessere Lesbarkeit der Topic-Terme.
    """
    df = df.copy()

    text_series = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
    if include_comments and "comments_text" in df.columns:
        comments_series = df["comments_text"].fillna("")
    else:
        comments_series = pd.Series([""] * len(df))

    df["text_all"] = (text_series.fillna("") + " " + comments_series).str.strip()

    df["clean"] = df["text_all"].map(clean_text)
    df["tokens_raw"] = df["clean"].map(tokenize)

    # Sprachheuristik
    df["lang"] = df["tokens_raw"].map(detect_language_by_stopword_ratio)

    # optional Englisch rausfiltern
    if filter_english:
        df = df[df["lang"] != "en"].reset_index(drop=True)

    # Stopwörter entfernen + optional Stemming
    def _proc(tokens):
        kept = []
        for t in tokens:
            if t in GERMAN_SW or t in ENGLISH_SW:
                continue
            if any(ch.isdigit() for ch in t):
                continue
            if len(t) < 3:
                continue
            if t in {"deleted"}:
                continue
            kept.append(t)
        return kept

    # >>> WICHTIG: tokens ERZEUGEN, bevor wir Kollokationen hinzufügen
    df["tokens"] = df["tokens_raw"].map(_proc)

    # Kollokations-Token ergänzen (PMI)
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
    def add_collocation_tokens(tokens_list, top_n=200, min_freq=5):
        all_tokens = [t for ts in tokens_list for t in ts]
        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(min_freq)
        scored = finder.score_ngrams(BigramAssocMeasures.pmi)
        top_bigrams = set(tuple(bg) for bg, _ in sorted(scored, key=lambda x: -x[1])[:top_n])

        new_list = []
        for ts in tokens_list:
            augmented = ts[:]
            for i in range(len(ts) - 1):
                pair = (ts[i], ts[i+1])
                if pair in top_bigrams:
                    augmented.append(f"{ts[i]}_{ts[i+1]}")
            new_list.append(augmented)
        return new_list

    df["tokens"] = add_collocation_tokens(df["tokens"], top_n=200, min_freq=5)
    df["clean_for_vect"] = df["tokens"].map(lambda ts: " ".join(ts))
    return df


def extract_flairs(df: pd.DataFrame) -> pd.Series:
    """
    Flair-Statistik; fällt tolerant auf leer zurück, wenn die CSV keine passende Spalte hat.
    """
    cand = [c for c in df.columns if c.lower() in ("flair_text", "link_flair_text", "flair")]
    if not cand:
        return pd.Series(dtype="int64")
    col = cand[0]
    return (df[col]
            .fillna("")
            .str.strip()
            .replace({"": None})
            .dropna()
            .value_counts())
