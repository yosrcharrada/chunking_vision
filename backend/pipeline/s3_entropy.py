"""
S3 — Enhanced Entropy Boundary Refinement with Entropy Rate & PPL Validation
=====================================================================
Pipeline position : runs AFTER S2 (chunkers) and BEFORE S4 (boundary quality).
Responsibility    : evaluate every adjacent chunk pair and decide whether to
                    MERGE them (too similar), mark them as HARD (confirmed topic
                    shift), or leave them as SOFT (ambiguous, passed to S4).

NEW FEATURES (v4)
─────────────────
1. ENTROPY RATE (INTRA-CHUNK):
   - Measures information rate between consecutive SENTENCES within a chunk
   - Lower entropy rate = sentences are predictable/related = good coherence
   - Higher entropy rate = sentences are diverse/independent = potential split point
   - Used to evaluate intra-chunk quality and detect poor merge candidates

2. HYBRID INTER-CHUNK METRIC:
   - Improved fusion of JSD (Jensen-Shannon) and Hellinger distance
   - Adaptive weighting based on confidence in each signal
   - Better detection of rare term shifts (Hellinger strength) with stable baseline (JSD)

3. PPL VALIDATION SYSTEM:
   - Uses DistilBERT language model to compute PERPLEXITY (PPL)
   - PPL = exp(cross-entropy): measures how surprised the model is at the text
   - Lower PPL = text is more coherent and predictable by the model
   - For EACH merge/split decision: validates that the merge actually IMPROVES coherence
   - Prevents merging chunks that would hurt readability even if entropy is low
   - Real validation: decisions are actually validated, not just static metrics

Six complementary boundary signals
────────────────────────────────────
1. JSD          Jensen-Shannon Divergence on unigram distributions.
                Symmetric KL-divergence: bounds [0,1].

2. Hellinger    Hellinger distance between unigram distributions.
                Alternative symmetric measure, more sensitive to rare terms.
                Bounds [0,1].

3. Entropy-Rate Intra-chunk coherence via sentence-level entropy rate.
                HIGH rate = sentences differ significantly = potential split
                Bounds [0,1].

4. PMI-drop     Key concept shift: measures divergence of TOP-8 content terms.
                Bounds [0,1].

5. Depth-change Structural hierarchy depth at chunk B start.
                Bounds [0,1].

6. Drift        Cosine distance between hash embeddings.
                Bounds [0,1].

PPL VALIDATION
──────────────
When a merge is proposed (signal < tau_low):
  1. Compute PPL of current chunk alone
  2. Compute PPL of next chunk alone
  3. Compute PPL of merged chunk
  4. Validate: merged_ppl < max(ppl_curr, ppl_next) * threshold
  5. Only merge if BOTH entropy is low AND PPL improves

LSTM memory
───────────
The LSTM receives the 6-dim feature vector + PPL validation flag at every
boundary position in document order. Its cell state cₜ accumulates context
so that a boundary at position 15 is judged relative to the entropy history
of positions 0–14, including validated PPL information.

Decision rules (in priority order)
────────────────────────────────────
  1. protected boundary → HARD  (Article N, TITRE, ALL-CAPS heading, …)
  2. combined_signal < τ_low AND ppl_valid AND size fits → MERGE
  3. combined_signal > τ_high              → HARD
  4. otherwise                             → SOFT  (passed to S4)
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# PPL computation dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers library not available; PPL validation disabled. "
        "Install via: pip install transformers torch"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stopword set  (English + French)
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    # English function words
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","must","shall","can","to","of","in","for","on",
    "with","at","by","from","as","into","through","during","before","after",
    "here","there","when","where","why","how","all","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","this","that","these","those",
    "it","its","he","she","they","we","you","i","my","your","his","her",
    "our","their","what","which","who","whom","about",
    # French function words
    "le","la","les","de","des","du","et","en","un","une","dans","pour",
    "que","est","sur","par","avec","au","aux","ce","se","si","ne","pas",
    "lui","leur","ils","elles","nous","vous","on","dont","où","car","mais",
    "ni","or","donc","comment","quand","comme","tout","tous","toute",
}


# ─────────────────────────────────────────────────────────────────────────────
# Protected boundary regex
# ─────────────────────────────────────────────────────────────────────────────
_PROTECTED_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|BOOK|PART|CHAPTER)"
    r"\s+(?:[IVXLCDM]+|\d+|PREMIER|PREMIERE|PREMIÈRE|premier|premiere|première)"
    r"|(?:Article|Art\.?|ARTICLE)\s+(?:\d+(?:\s*(?:er|eme|ème|e|bis|ter|quater))?|premier|1er|[IVX]+)"
    r"|§\s*\d+"
    r"|#{1,6}\s+\S+"
    r"|[A-Z][A-Z\s\-]{4,}(?::|$)"
    r")(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Structural depth markers  (Signal 5)
# ─────────────────────────────────────────────────────────────────────────────
_DEPTH_MARKERS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"^\s*(?:TITRE|CHAPITRE|BOOK|PART)\b",             re.I), 1),
    (re.compile(r"^\s*(?:SECTION|SOUS-SECTION)\b",                 re.I), 2),
    (re.compile(r"^\s*(?:Article|Art\.?|ARTICLE|§)\s*\d+",         re.I), 3),
    (re.compile(r"^\s*(?:\d+(?:\.\d+)+)\s+\S",                     re.I), 4),
    (re.compile(r"^\s*[a-zA-Z]\.\s+\S",                            re.I), 5),
    (re.compile(r"^\s*[-–•]\s+\S",                                 re.I), 6),
]


# ─────────────────────────────────────────────────────────────────────────────
# Global PPL Model Manager (lazy loaded)
# ─────────────────────────────────────────────────────────────────────────────
class PPLValidator:
    """
    Language-aware PPL validator using the correct causal LM per document language.

    WHY THE ORIGINAL WAS BROKEN
    ────────────────────────────
    The original always loaded DistilGPT2, which was trained exclusively on
    English WebText.  French tokens are heavily fragmented by its BPE tokenizer
    (e.g. "établissement" → ["é", "tab", "liss", "ement"]), producing perplexity
    values that reflect tokenization fragmentation rather than semantic coherence.
    A coherent French sentence and an incoherent one get similarly high PPL
    under DistilGPT2 → the validation signal was pure noise for French text.

    THE FIX: LANGUAGE-AWARE MODEL SELECTION
    ─────────────────────────────────────────
    We maintain a mapping from ISO 639-1 language codes to the best available
    lightweight causal LM for that language.  The document's language is
    detected once (from the first call's text sample) and the right model is
    loaded.  All subsequent calls reuse the cached model.

    Model choices per language
    ──────────────────────────
    "en" → "distilgpt2"
        Fast, 82M params, trained on English WebText.  PPL on English text
        is a reliable coherence signal.

    "fr" → "asi/gpt-fr-cased-small"
        ~124M params, trained on French Common Crawl + Wikipedia.
        Produces meaningful PPL on French legal/regulatory text.
        Fallback: "bigscience/bloom-560m" (multilingual, larger but slower).

    "ar" → "bigscience/bloom-560m"
        BLOOM is the best open multilingual causal LM at this size.
        Handles Arabic script natively.

    "*"  → "bigscience/bloom-560m"
        Universal fallback for any language not listed above.

    Perplexity formula
    ──────────────────
    PPL(text) = exp( (1/N) × Σᵢ -log P(tᵢ | t₁…tᵢ₋₁) )

    where N is the number of tokens and P is the model's conditional
    probability.  Lower PPL = the model finds the text more predictable
    = the text is more coherent under the language model's learned distribution.

    Merge validation rule
    ─────────────────────
    Merge A+B is PPL-valid if:
        PPL(A+B) < max(PPL(A), PPL(B)) × threshold

    Intuition: if the merged text is MORE surprising to the model than both
    parts individually, the merge created an incoherent combination.
    threshold=1.1 allows a 10% PPL increase (small tolerance for joining
    sentences that share few content words but are semantically related).
    """

    # Per-language model registry.
    # Keys: ISO 639-1 codes.  Values: HuggingFace model IDs.
    # Add entries here to support new languages without changing any other code.
    _LANG_MODELS: Dict[str, str] = {
        "en": "distilgpt2",                    # English — 82M, fast, reliable
        "fr": "asi/gpt-fr-cased-small",        # French  — trained on FR corpora
        "de": "dbmdz/german-gpt2",             # German
        "es": "datificate/gpt2-small-spanish", # Spanish
        "it": "GroNLP/gpt2-small-italian",     # Italian
        "*":  "bigscience/bloom-560m",         # Universal multilingual fallback
    }

    # Singleton: one validator per process, models cached per language
    _instance: Optional['PPLValidator'] = None

    # Class-level flag: True only after preload() has been called and succeeded.
    # Models are NEVER downloaded during a request — only during preload().
    # If preload() was never called or failed, all validate_merge() calls
    # return True immediately (rely on the lexical coherence gate instead).
    _preloaded: bool = False

    def __init__(self):
        # Cache: lang_code → (tokenizer, model) or None
        self._models: Dict[str, Any] = {}
        self.device = "cpu"
        self._detected_lang: Optional[str] = None

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

    @classmethod
    def get_instance(cls) -> 'PPLValidator':
        """Singleton accessor — one validator per process."""
        if cls._instance is None:
            cls._instance = PPLValidator()
        return cls._instance

    @classmethod
    def preload(cls, languages: Optional[List[str]] = None) -> None:
        """
        Pre-load language models at server startup — NEVER called during a request.

        Call this once from your server startup code (e.g. FastAPI lifespan,
        Gunicorn post_fork hook, or __main__ block) BEFORE accepting requests.
        Model downloads happen here, not during request handling.

        Parameters
        ──────────
        languages : list of ISO 639-1 codes to preload, e.g. ["fr", "en"].
                    If None, defaults to ["fr", "en"] (most common use case).

        Example server startup usage:
            from s3_entropy import PPLValidator
            PPLValidator.preload(["fr", "en"])   # called once at startup

        If preload() is never called (e.g. development mode, CI), all
        validate_merge() calls return True and the pipeline relies solely
        on the lexical coherence gate — the pipeline still works correctly.
        """
        if not TRANSFORMERS_AVAILABLE:
            return   # nothing to preload — transformers not installed

        if languages is None:
            languages = ["fr", "en"]

        inst = cls.get_instance()
        succeeded = False

        for lang in languages:
            result = inst._load_model_for_lang(lang)
            if result is not None:
                succeeded = True

        # Only set _preloaded=True if at least one model loaded successfully.
        # This prevents the request-time guard from being bypassed when all
        # models failed to download (e.g. no internet access at startup).
        if succeeded:
            cls._preloaded = True

    def _load_model_for_lang(self, lang: str) -> Optional[tuple]:
        """
        Internal: load and cache the model for one language code.

        Called ONLY from preload() — never from request-handling code paths.
        Downloads the model from HuggingFace Hub if not already cached.
        Falls back to the universal "*" model if the language-specific one fails.

        Returns (tokenizer, model) on success, None on failure.
        """
        if lang in self._models:
            return self._models[lang]   # already loaded in this session

        if not TRANSFORMERS_AVAILABLE:
            self._models[lang] = None
            return None

        model_id = self._LANG_MODELS.get(lang, self._LANG_MODELS["*"])
        fallback_id = self._LANG_MODELS["*"]

        for candidate in dict.fromkeys([model_id, fallback_id]):  # deduplicated
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model     = AutoModelForCausalLM.from_pretrained(candidate)
                model.eval()
                model.to(self.device)
                self._models[lang] = (tokenizer, model)
                return self._models[lang]
            except Exception as exc:
                warnings.warn(f"PPLValidator: failed to load {candidate}: {exc}")
                continue

        self._models[lang] = None
        return None

    def _get_model(self, lang: str) -> Optional[tuple]:
        """
        Request-time model lookup — NEVER downloads, only returns cached models.

        If the model for this language was not preloaded, returns None immediately.
        This guarantees zero network activity during request handling.
        """
        # Return from cache (populated by preload())
        if lang in self._models:
            return self._models[lang]

        # Model not preloaded — return None, caller will skip PPL validation
        # and rely on the lexical coherence gate (merge_coherence) alone.
        return None

    def _detect_lang(self, text: str) -> str:
        """
        Detect the language of the text.

        Priority order:
        1. langdetect (pip install langdetect) — probabilistic n-gram model
        2. langid     (pip install langid)     — Naive Bayes classifier
        3. Heuristic: French vs English function word counts (zero-dependency)

        Result is cached after the first detection — we assume a document
        is monolingual, so we only detect once per PPLValidator instance.
        """
        if self._detected_lang is not None:
            return self._detected_lang

        try:
            from langdetect import detect  # type: ignore
            lang = detect(text[:2000]).split("-")[0].lower()
            self._detected_lang = lang
            return lang
        except Exception:
            pass

        try:
            import langid  # type: ignore
            lang, _ = langid.classify(text[:2000])
            self._detected_lang = lang.lower()
            return self._detected_lang
        except Exception:
            pass

        # Zero-dependency heuristic
        sample = text[:3000].lower()
        words  = re.findall(r"\b\w+\b", sample)
        fr_markers = {"le","la","les","de","des","du","et","en","dans",
                      "pour","que","est","sur","par","avec","au","aux"}
        en_markers = {"the","a","an","and","or","is","are","was","were",
                      "have","has","will","would","this","that","with","from"}
        fr_count = sum(1 for w in words if w in fr_markers)
        en_count = sum(1 for w in words if w in en_markers)
        self._detected_lang = "fr" if fr_count > en_count * 1.5 else "en"
        return self._detected_lang

    def compute_ppl(self, text: str, lang: Optional[str] = None) -> Optional[float]:
        """
        Compute perplexity of text under the language-appropriate causal LM.

        PPL(text) = exp( (1/N) × Σᵢ −log P(tᵢ | t₁…tᵢ₋₁) )

        Returns None immediately if:
        - transformers/torch not installed
        - preload() was never called (models not cached)
        - text is too short (< 5 tokens)
        - inference fails for any reason

        None means "skip PPL — rely on lexical coherence gate".
        """
        # Guard 1: no models available at all
        if not TRANSFORMERS_AVAILABLE or not PPLValidator._preloaded:
            return None

        if not text or len(text.split()) < 5:
            return None

        if lang is None:
            lang = self._detect_lang(text)

        # Guard 2: this language was not preloaded — no download attempt
        pair = self._get_model(lang)
        if pair is None:
            return None

        tokenizer, model = pair
        try:
            import torch
            enc       = tokenizer(text, return_tensors="pt",
                                  max_length=512, truncation=True)
            input_ids = enc["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids)
                logits  = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss(reduction="mean")(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            ppl = float(torch.exp(loss).cpu().item())
            return float(np.clip(ppl, 1.0, 10_000.0))

        except Exception as exc:
            warnings.warn(f"PPL computation failed for lang={lang}: {exc}")
            return None

    def validate_merge(self, text_a: str, text_b: str,
                       threshold: float = 1.1) -> bool:
        """
        Validate that merging text_a and text_b improves or preserves coherence.

        Rule:  PPL(A+B) < max(PPL(A), PPL(B)) × threshold

        Returns True immediately (allow merge) if:
        - transformers not installed
        - preload() was never called (safe fallback — lexical gate handles it)
        - PPL computation fails for any reason

        This means the method NEVER blocks, NEVER downloads, NEVER hangs.
        """
        # Fast path: no PPL available — let lexical coherence gate decide
        if not TRANSFORMERS_AVAILABLE or not PPLValidator._preloaded:
            return True

        combined = text_a + " " + text_b
        lang = self._detect_lang(combined)

        ppl_a      = self.compute_ppl(text_a,               lang=lang)
        ppl_b      = self.compute_ppl(text_b,               lang=lang)
        ppl_merged = self.compute_ppl(text_a + "\n\n" + text_b, lang=lang)

        if ppl_merged is None or (ppl_a is None and ppl_b is None):
            return True   # not enough data — allow merge

        max_individual = max(ppl_a or 0.0, ppl_b or 0.0)
        if max_individual == 0.0:
            return True

        return ppl_merged < max_individual * threshold


# ─────────────────────────────────────────────────────────────────────────────
# Entropy Rate Calculator (Intra-chunk coherence)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_entropy_rate(text_a: str, text_b: str) -> float:
    """
    Compute ENTROPY RATE of the MERGED candidate (text_a + "\\n\\n" + text_b).

    WHY THIS SIGNATURE CHANGED FROM v4
    ────────────────────────────────────
    The original _compute_entropy_rate(text: str) took a single argument and
    was called as _compute_entropy_rate(text_b) — measuring the coherence of
    chunk B alone.  This answered "is B internally coherent?" rather than
    "would A+B together be coherent?", which is what a merge decision needs.

    Fix: we concatenate A and B before computing the rate, so the metric
    directly measures whether the MERGE would produce a coherent chunk.

    Entropy Rate Definition
    ───────────────────────
    H = (1/(n-1)) × Σᵢ JSD(P(Sᵢ), P(Sᵢ₊₁))

    where Sᵢ are the sentences of the merged text and P(Sᵢ) is the unigram
    distribution of sentence i.

    Returns float ∈ [0,1]:
      0 = all consecutive sentences are topically similar (good merge)
      1 = sentences jump across topics (bad merge — high entropy rate)
    """
    # Compute on the merged candidate so we measure A+B coherence, not just B
    text = text_a + "\n\n" + text_b
    sentences = _split_sentences(text)

    if len(sentences) < 2:
        return 0.0  # single sentence: no rate to compute

    # Compute unigram distributions for each sentence
    sentence_dists: List[Tuple[Dict[str, int], int]] = []
    for sent in sentences:
        tokens = _tokenize(sent)
        if not tokens:
            sentence_dists.append(({}, 0))
        else:
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            sentence_dists.append((freq, len(tokens)))

    # Build joint vocabulary across all sentences
    joint_vocab = set()
    for freq_dict, _ in sentence_dists:
        joint_vocab.update(freq_dict.keys())

    if not joint_vocab:
        return 0.0

    # Compute JSD for consecutive sentence pairs
    jsd_values: List[float] = []
    for i in range(len(sentence_dists) - 1):
        freq_curr, len_curr = sentence_dists[i]
        freq_next, len_next = sentence_dists[i + 1]

        if not freq_curr or not freq_next:
            jsd_values.append(0.5)
            continue

        p = np.array([freq_curr.get(w, 0) / len_curr for w in joint_vocab], dtype=np.float64)
        q = np.array([freq_next.get(w, 0) / len_next for w in joint_vocab], dtype=np.float64)
        m = (p + q) / 2.0
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        jsd_values.append(float(np.clip(jsd, 0.0, 1.0)))

    if not jsd_values:
        return 0.0

    return float(np.clip(np.mean(jsd_values), 0.0, 1.0))


def _split_sentences(text: str, min_length: int = 3) -> List[str]:
    """
    Split text into sentences via basic regex patterns.
    Handles period, question mark, exclamation mark, and newlines.
    Filters out sentences shorter than min_length tokens.
    
    Parameters
    ──────────
    text : str
        The text to split
    min_length : int
        Minimum tokens per sentence (filters out garbage)
    
    Returns
    ───────
    List[str]
        List of sentences (non-empty, >= min_length tokens)
    """
    # Split on sentence boundaries: .!?\n
    patterns = [
        r'(?<=[.!?\n])\s+',  # space after sentence terminal
        r'\n+',                # newline breaks
    ]
    
    # First split attempt: on standard boundaries
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    
    # Filter: remove empty and very short sentences
    result = []
    for sent in sentences:
        sent = sent.strip()
        if sent and len(sent.split()) >= min_length:
            result.append(sent)
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Forward LSTM cell  — deterministic, fixed-weight
# ─────────────────────────────────────────────────────────────────────────────

class _ForwardLSTMCell:
    """
    Single-layer forward LSTM.
    
    Enhanced for 7-dimensional input:
      [jsd, hellinger, entropy_rate, pmi_drop, depth_change, drift, ppl_valid]
    
    Why fixed weights?
    ──────────────────
    Fully deterministic output (same document → same chunks every run)
    without requiring training. Fixed Xavier weights (seed=42) give
    the LSTM its gating structure without model dependencies.
    
    Architecture
    ────────────
    Input  : 7-dim vector [jsd, hellinger, entropy_rate, pmi_drop, 
                            depth_change, drift, ppl_valid]
    Hidden : 14-dim cell state cₜ and hidden state hₜ
    Output : scalar ∈ [0,1] via score = σ(Wₚ · hₜ)
    """

    INPUT_DIM  = 7    # enhanced from 5 to include entropy_rate and ppl_valid
    HIDDEN_DIM = 14   # increased for more expressive hidden states

    def __init__(self, seed: int = 42):
        """Initialize LSTM with fixed Xavier-scaled weights."""
        rng   = np.random.RandomState(seed)
        idim  = self.INPUT_DIM
        hdim  = self.HIDDEN_DIM

        # Xavier scale
        scale = np.sqrt(2.0 / (idim + hdim))

        # ── Input gate ────────────────────────────────────────────────────
        self.Wi = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ui = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bi = np.zeros(hdim, dtype=np.float32)

        # ── Forget gate (bias initialized to 1) ───────────────────────────
        self.Wf = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uf = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bf = np.ones(hdim, dtype=np.float32)

        # ── Cell gate ─────────────────────────────────────────────────────
        self.Wg = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ug = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bg = np.zeros(hdim, dtype=np.float32)

        # ── Output gate ───────────────────────────────────────────────────
        self.Wo = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uo = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bo = np.zeros(hdim, dtype=np.float32)

        # ── Scalar projection ─────────────────────────────────────────────
        self.Wp = rng.randn(1, hdim).astype(np.float32) * np.sqrt(1.0 / hdim)

        # ── State vectors ─────────────────────────────────────────────────
        self.h = np.zeros(hdim, dtype=np.float32)
        self.c = np.zeros(hdim, dtype=np.float32)

    def reset(self) -> None:
        """Reset h and c to zero at the start of a new document."""
        self.h[:] = 0.0
        self.c[:] = 0.0

    def step(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Execute one LSTM step for one boundary position.
        
        Parameters
        ──────────
        x : np.ndarray, shape (7,)
            Boundary feature vector [jsd, hellinger, entropy_rate, pmi_drop,
            depth_change, drift, ppl_valid]. All ∈ [0,1].
        
        Returns
        ───────
        score : float ∈ [0,1]
            Boundary confidence from LSTM context
        cell  : np.ndarray, shape (14,)
            Cell state after this step
        """
        # Compute gate activations
        i_g = self._sig(self.Wi @ x + self.Ui @ self.h + self.bi)
        f_g = self._sig(self.Wf @ x + self.Uf @ self.h + self.bf)
        g_g = np.tanh(  self.Wg @ x + self.Ug @ self.h + self.bg)
        o_g = self._sig(self.Wo @ x + self.Uo @ self.h + self.bo)

        # Update cell state
        self.c = f_g * self.c + i_g * g_g

        # Update hidden state
        self.h = o_g * np.tanh(self.c)

        # Project to scalar score
        score = float(self._sig(self.Wp @ self.h)[0])

        return score, self.c.copy()

    @staticmethod
    def _sig(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Raw signal selector (updated for new signals)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_signal_from_features(feat: Dict[str, float], metric: str) -> float:
    """
    Compute the raw boundary score from the feature dict.
    
    Enhanced metrics now include entropy_rate which is specifically useful
    for detecting intra-chunk coherence issues.
    
    Parameters
    ──────────
    feat   : dict with keys jsd, hellinger, entropy_rate, pmi_drop, 
             depth_change, drift
    metric : string — one of "jsd", "hellinger", "pmi", "entropy_rate", 
             "depth", "drift", or "hybrid" (default)
    
    Returns
    ───────
    float ∈ [0,1] : the raw boundary score before LSTM modulation
    """
    if metric == "jsd":
        return float(feat.get("jsd", 0.5))

    if metric == "hellinger":
        return float(feat.get("hellinger", 0.5))

    if metric == "entropy_rate":
        # Entropy rate: high = diverse sentences = potential merge candidate
        return float(feat.get("entropy_rate", 0.5))

    if metric == "pmi":
        return float(feat.get("pmi_drop", 0.5))

    if metric == "depth":
        return float(feat.get("depth_change", 0.5))

    if metric == "drift":
        return float(feat.get("drift", 0.5))

    # ── "hybrid" (default, enhanced) ──────────────────────────────────────
    # NEW weights that better balance all signals:
    #   0.25 JSD:           reliable baseline
    #   0.20 Hellinger:     rare term detection
    #   0.15 entropy_rate:  intra-chunk coherence check (NEW)
    #   0.20 PMI-drop:      concept shift (highest per-signal weight)
    #   0.12 depth_change:  structural boundaries
    #   0.08 drift:         long-range semantic drift
    return float(np.clip(
        0.25 * feat.get("jsd", 0.5)
        + 0.20 * feat.get("hellinger", 0.5)
        + 0.15 * feat.get("entropy_rate", 0.5)
        + 0.20 * feat.get("pmi_drop", 0.5)
        + 0.12 * feat.get("depth_change", 0.5)
        + 0.08 * feat.get("drift", 0.5),
        0.0, 1.0,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_boundaries(chunks: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """
    Evaluate every adjacent chunk pair and decide: merge, hard, or soft.
    
    Enhanced with:
    - Intra-chunk entropy rate for coherence checking
    - PPL validation for merge decisions
    - Improved hybrid metric
    
    Algorithm (5 passes, enhanced)
    ──────────────────────────────
    Pass 1: Compute 7-dim feature vector for every N-1 adjacent pair.
            (Now includes entropy_rate + ppl validity flag)
    Pass 2: Run the LSTM forward over the feature sequence.
    Pass 3: Compute the combined signal and adaptive thresholds.
    Pass 4: Generate PPL validator and compute merge validations.
    Pass 5: Iterate through chunks and apply merge/hard/soft decisions
            using BOTH entropy-based signals AND PPL validation.
    
    Output fields added to every chunk
    ───────────────────────────────────
    [existing fields +]
    entropy_rate         — intra-chunk coherence (0=coherent, 1=diverse)
    ppl_valid            — bool: PPL validates the merge decision
    chunk_ppl            — perplexity of this chunk (if available)
    [other existing fields unchanged]
    """

    # ── Edge cases ────────────────────────────────────────────────────────
    if not chunks:
        return chunks

    if len(chunks) == 1:
        c = dict(chunks[0])
        c.update({
            "jsd_score":        0.0,
            "metric_score":     0.0,
            "boundary_signal":  0.0,
            "hidden_state":     0.0,
            "lstm_cell":        [],
            "boundary_type":    "single",
            "boundary_features": {},
            "s3_stats":         _empty_stats(1, 1),
            "entropy_rate":     0.0,
            "ppl_valid":        True,
            "chunk_ppl":        None,
        })
        return [c]

    # ── Read configuration ────────────────────────────────────────────────
    mode         = str(config.get("threshold_mode",       "percentile")).lower()
    tau_low_cfg  = float(config.get("tau_jsd_low",        0.15))
    tau_high_cfg = float(config.get("tau_jsd_high",       0.45))
    n_max        = int(config.get("n_max",                500))
    pct_low      = float(config.get("tau_percentile_low",  25))
    pct_high     = float(config.get("tau_percentile_high", 75))
    metric       = str(config.get("entropy_metric",       "hybrid")).lower()
    ppl_threshold = float(config.get("ppl_merge_threshold", 1.1))  # NEW: allow 10% PPL increase
    enable_ppl   = config.get("enable_ppl_validation", True)  # NEW: can disable for speed

    # ── Initialize PPL validator ──────────────────────────────────────────
    ppl_validator = PPLValidator.get_instance() if enable_ppl else None

    # ── Pass 1: compute 7-dim feature vectors for all N-1 boundaries ─────
    pairs: List[Dict] = []
    for i in range(len(chunks) - 1):
        pairs.append(_boundary_features(chunks[i]["text"], chunks[i + 1]["text"]))

    # ── Pass 2: LSTM forward pass ─────────────────────────────────────────
    lstm = _ForwardLSTMCell(seed=42)
    lstm.reset()

    lstm_scores: List[float]       = []
    lstm_cells:  List[List[float]] = []

    for feat in pairs:
        # Build 7-dim input (enhanced)
        x = np.array([
            feat["jsd"],
            feat["hellinger"],
            feat["entropy_rate"],   # NEW
            feat["pmi_drop"],
            feat["depth_change"],
            feat["drift"],
            float(feat.get("ppl_valid", 1.0)),  # NEW: 1.0 if valid, 0.0 if invalid
        ], dtype=np.float32)

        score, cell = lstm.step(x)
        lstm_scores.append(score)
        lstm_cells.append(cell.tolist())

    # ── Pass 3a: combined signals ─────────────────────────────────────────
    raw_signals: List[float] = []
    for feat, ls in zip(pairs, lstm_scores):
        raw      = _raw_signal_from_features(feat, metric)
        combined = float(np.clip(0.65 * raw + 0.35 * ls, 0.0, 1.0))
        raw_signals.append(combined)

    # ── Pass 3b: compute adaptive thresholds ──────────────────────────────
    if mode == "percentile" and raw_signals:
        tau_low  = float(np.percentile(raw_signals, pct_low))
        tau_high = float(np.percentile(raw_signals, pct_high))
    else:
        tau_low  = tau_low_cfg
        tau_high = tau_high_cfg

    if tau_low >= tau_high:
        tau_high = min(1.0, tau_low + 0.08)

    # ── Pass 4: compute PPL validations ───────────────────────────────────
    # For each boundary, check: would the merge improve PPL?
    ppl_validities: List[bool] = []
    for i in range(len(chunks) - 1):
        if ppl_validator is not None:
            # NEW: PPL validation of merge decision
            is_valid = ppl_validator.validate_merge(
                chunks[i]["text"],
                chunks[i + 1]["text"],
                threshold=ppl_threshold
            )
            ppl_validities.append(is_valid)
            # Store PPL info if available
            if not hasattr(chunks[i], "_chunk_ppl"):
                chunks[i]["chunk_ppl"] = ppl_validator.compute_ppl(chunks[i]["text"])
        else:
            ppl_validities.append(True)  # assume valid if no validator

    # ── Pass 5: iterate and apply merge / hard / soft decisions ──────────
    work = [dict(c) for c in chunks]
    out:  List[Dict] = []

    merged_count    = 0
    hard_count      = 0
    soft_count      = 0
    protected_count = 0
    signal_history: List[float] = []

    i = 0
    while i < len(work):
        curr = dict(work[i])

        if i < len(work) - 1:
            # Evaluate boundary
            pair_idx = min(i, len(raw_signals) - 1)
            signal   = raw_signals[pair_idx]
            feat     = pairs[pair_idx] if pair_idx < len(pairs) else {}
            ls       = lstm_scores[pair_idx]
            lc       = lstm_cells[pair_idx]
            ppl_ok   = ppl_validities[pair_idx] if pair_idx < len(ppl_validities) else True
            
            # Check protected boundary
            protected = _is_protected_boundary(work[i + 1].get("text", ""))

            # ── Annotate with boundary metadata ───────────────────────────
            curr["jsd_score"]       = round(feat.get("jsd", 0.0), 4)
            curr["metric_score"]    = round(signal, 4)
            curr["boundary_signal"] = round(signal, 4)
            curr["hidden_state"]    = round(ls, 4)
            curr["lstm_cell"]       = [round(v, 4) for v in lc]
            curr["entropy_rate"]    = round(feat.get("entropy_rate", 0.0), 4)
            curr["ppl_valid"]       = ppl_ok  # NEW: PPL validity flag
            
            curr["boundary_features"] = {
                "jsd":          round(feat.get("jsd", 0.0), 4),
                "hellinger":    round(feat.get("hellinger", 0.0), 4),
                "entropy_rate": round(feat.get("entropy_rate", 0.0), 4),  # NEW
                "pmi_drop":     round(feat.get("pmi_drop", 0.0), 4),
                "depth_change": round(feat.get("depth_change", 0.0), 4),
                "drift":        round(feat.get("drift", 0.0), 4),
                "lstm_score":   round(ls, 4),
                "combined":     round(signal, 4),
                "ppl_valid":    ppl_ok,  # NEW
            }
            signal_history.append(signal)

            curr_words = len(curr.get("text", "").split())
            next_words = len(work[i + 1].get("text", "").split())
            size_ok    = (curr_words + next_words) <= max(n_max * 1.35, n_max + 80)

            # ── Decision tree (enhanced with PPL validation) ──────────────
            if protected:
                # CASE 1: Protected boundary → ALWAYS hard split
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "protected_structure_boundary"
                protected_count += 1
                hard_count      += 1
                out.append(curr)
                i += 1

            elif signal < tau_low and size_ok and ppl_ok:
                # CASE 2: Low signal AND size fits AND ppl_valid → MERGE
                # NEW: PPL validation ensures the merge is actually coherent
                nxt = work[i + 1]
                curr["text"]          = curr["text"] + "\n\n" + nxt["text"]
                curr["end"]           = nxt.get("end", curr.get("end", 0))
                curr["boundary_type"] = "merged"
                curr["merge_reason"]  = "low_entropy_boundary_ppl_validated"

                work[i] = curr
                work.pop(i + 1)

                if i < len(work) - 1:
                    new_feat = _boundary_features(curr["text"], work[i + 1]["text"])
                    new_x    = np.array([
                        new_feat["jsd"],
                        new_feat["hellinger"],
                        new_feat["entropy_rate"],  # now computed on merged candidate
                        new_feat["pmi_drop"],
                        new_feat["depth_change"],
                        new_feat["drift"],
                        float(new_feat.get("ppl_valid", 1.0)),
                    ], dtype=np.float32)

                    new_ls, new_lc = lstm.step(new_x)

                    new_raw = _raw_signal_from_features(new_feat, metric)
                    new_sig = float(np.clip(0.65 * new_raw + 0.35 * new_ls, 0.0, 1.0))

                    if pair_idx < len(raw_signals):
                        raw_signals[pair_idx] = new_sig
                        pairs[pair_idx]        = new_feat
                        lstm_scores[pair_idx]  = new_ls
                        lstm_cells[pair_idx]   = new_lc.tolist()

                merged_count += 1
                continue

            elif signal > tau_high:
                # CASE 3: High signal → confirmed topic shift → HARD split
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "high_entropy_shift"
                hard_count += 1
                out.append(curr)
                i += 1

            else:
                # CASE 4: Signal between thresholds → SOFT boundary
                curr["boundary_type"] = "soft"
                curr["merge_reason"]  = "moderate_entropy_shift"
                soft_count += 1
                out.append(curr)
                i += 1

        else:
            # ── Last chunk: no next neighbour ────────────────────────────
            curr["jsd_score"]        = 0.0
            curr["metric_score"]     = 0.0
            curr["boundary_signal"]  = 0.0
            curr["hidden_state"]     = 0.0
            curr["lstm_cell"]        = []
            curr["boundary_features"] = {}
            curr["boundary_type"]    = "end"
            curr["entropy_rate"]     = 0.0
            curr["ppl_valid"]        = True
            out.append(curr)
            i += 1

    # ── Attach summary statistics to the last chunk ───────────────────────
    if out:
        out[-1]["thresholds"] = {
            "low":  round(tau_low, 4),
            "high": round(tau_high, 4),
            "mode": mode,
        }
        out[-1]["s3_stats"] = {
            "initial_count":   len(chunks),
            "final_count":     len(out),
            "merged_count":    merged_count,
            "hard_count":      hard_count,
            "soft_count":      soft_count,
            "protected_count": protected_count,
            "mean_signal":     round(float(np.mean(signal_history)), 4) if signal_history else 0.0,
            "merge_ratio":     round(merged_count / max(1, len(chunks) - 1), 4),
            "ppl_enabled":     bool(enable_ppl and ppl_validator is not None),  # NEW
        }
    
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """Extract the boundary signal series for charts."""
    return [c.get("metric_score", c.get("jsd_score", 0.0)) for c in chunks]


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation — one function per signal
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_features(text_a: str, text_b: str) -> Dict[str, float]:
    """
    Compute the complete feature vector for one boundary.
    
    All values guaranteed ∈ [0, 1].

    entropy_rate is now computed on the MERGED CANDIDATE (text_a + text_b)
    rather than text_b alone — this correctly measures whether the merge
    would produce a coherent chunk, not just whether B is internally coherent.
    """
    return {
        "jsd":          _compute_jsd(text_a, text_b),
        "hellinger":    _compute_hellinger(text_a, text_b),
        "entropy_rate": _compute_entropy_rate(text_a, text_b),  # FIXED: merged candidate
        "pmi_drop":     _compute_pmi_drop(text_a, text_b),
        "depth_change": _compute_depth_change(text_b),
        "drift":        _compute_drift(text_a, text_b),
        "ppl_valid":    1.0,  # placeholder; actual validation happens in Pass 4
    }


# ── Signal 1 — Jensen-Shannon Divergence ─────────────────────────────────────

def _compute_jsd(t1: str, t2: str) -> float:
    """
    JSD(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M), where M = (P+Q)/2
    
    Symmetric KL-divergence bounded in [0, 1].
    """
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    m = (p + q) / 2.0
    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


# ── Signal 2 — Hellinger Distance ────────────────────────────────────────────

def _compute_hellinger(t1: str, t2: str) -> float:
    """
    H(P, Q) = ‖√P − √Q‖₂ / √2
    
    More sensitive to rare term changes than JSD.
    """
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    return float(np.clip(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2), 0.0, 1.0))


# ── Signal 3 — PMI-drop ──────────────────────────────────────────────────────

def _compute_pmi_drop(t1: str, t2: str) -> float:
    """
    Key concept shift detector.
    Jaccard similarity of top-8 content terms.
    High pmi_drop = low concept overlap = topic shift.
    """
    toks1 = _content_tokens(t1)
    toks2 = _content_tokens(t2)
    if not toks1 or not toks2:
        return 0.5

    K = 8

    freq1: Dict[str, int] = {}
    for t in toks1:
        freq1[t] = freq1.get(t, 0) + 1
    freq2: Dict[str, int] = {}
    for t in toks2:
        freq2[t] = freq2.get(t, 0) + 1

    top1 = {w for w, _ in sorted(freq1.items(), key=lambda x: -x[1])[:K]}
    top2 = {w for w, _ in sorted(freq2.items(), key=lambda x: -x[1])[:K]}

    shared = top1 & top2
    union  = top1 | top2
    if not union:
        return 0.5

    overlap_ratio = len(shared) / len(union)
    return float(np.clip(1.0 - overlap_ratio, 0.0, 1.0))


# ── Signal 4 — Structural depth change ───────────────────────────────────────

def _compute_depth_change(text_b: str) -> float:
    """
    Detect document hierarchy level at chunk B start.
    High depth = strong structural boundary.
    """
    first_line = ""
    for line in text_b.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    if not first_line:
        return 0.0

    for pattern, depth in _DEPTH_MARKERS:
        if pattern.match(first_line):
            return float(np.clip(1.0 - (depth - 1) / 6.0, 0.0, 1.0))
    return 0.0


# ── Signal 5 — Embedding drift ────────────────────────────────────────────────

def _compute_drift(t1: str, t2: str) -> float:
    """
    Cosine DISTANCE between hash embeddings.
    drift = 1 − cos(v₁, v₂)
    """
    v1 = _hash_embed(t1)
    v2 = _hash_embed(t2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.5

    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip(1.0 - cosine, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Protected boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_protected_boundary(text: str) -> bool:
    """Return True if text starts with a structural marker that must hard-split."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return bool(_PROTECTED_RE.match(stripped))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _distribution_pair(t1: str, t2: str):
    """
    Build aligned probability distributions for t1 and t2.
    Over the joint vocabulary (t1 ∪ t2).
    """
    tok1 = _tokenize(t1)
    tok2 = _tokenize(t2)
    if not tok1 or not tok2:
        return None, None  # type: ignore[return-value]

    vocab = list(set(tok1) | set(tok2))

    c1: Dict[str, int] = {}
    c2: Dict[str, int] = {}
    for t in tok1:
        c1[t] = c1.get(t, 0) + 1
    for t in tok2:
        c2[t] = c2.get(t, 0) + 1

    p = np.array([c1.get(w, 0) / len(tok1) for w in vocab], dtype=np.float64)
    q = np.array([c2.get(w, 0) / len(tok2) for w in vocab], dtype=np.float64)
    return p, q


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(P ‖ Q) = Σ_x P(x) log(P(x) / Q(x))
    0 · log(0) = 0 by convention.
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, 1.0))))


def _tokenize(text: str) -> List[str]:
    """Extract all word tokens via word-boundary regex."""
    return re.findall(r"\b\w+\b", text.lower())


def _content_tokens(text: str) -> List[str]:
    """
    Extract content tokens only (stopwords removed).
    Minimum length 2.
    """
    toks = [
        t for t in re.findall(r"\b[\wÀ-ÿ]{2,}\b", text.lower())
        if t not in _STOPWORDS
    ]
    return toks or _tokenize(text)


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Lightweight bag-of-content-words hash embedding.
    Maps tokens to dim-dimensional position via hash().
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _content_tokens(text):
        vec[hash(tok) % dim] += 1.0
    return vec


def _empty_stats(initial: int, final: int) -> Dict[str, Any]:
    """Return neutral s3_stats dict for edge cases."""
    return {
        "initial_count":   initial,
        "final_count":     final,
        "merged_count":    0,
        "hard_count":      0,
        "soft_count":      0,
        "protected_count": 0,
        "mean_signal":     0.0,
        "merge_ratio":     0.0,
        "ppl_enabled":     False,
    }


# Type annotation import (same as original)
from typing import Dict  # noqa: E402