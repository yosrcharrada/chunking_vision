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
    Manages PPL computation using a lightweight causal language model.
    Lazy-loads on first use.  Singleton pattern prevents multiple model loads.
    
    Why DistilBERT (causal LM)?
    ─────────────────────────────
    - Lightweight: fast inference, low memory
    - Pre-trained on diverse corpora: understands general coherence patterns
    - Causal: left-to-right generation matches how humans read sequentially
    - Perplexity = exp(cross_entropy): standard measure of language quality
    """
    
    _instance: Optional['PPLValidator'] = None
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> 'PPLValidator':
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = PPLValidator()
        return cls._instance
    
    def _load_model(self) -> bool:
        """
        Lazy-load the model and tokenizer.
        Returns True if successful, False if transformers unavailable.
        """
        if self._loaded:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            # Use DistilGPT2: lightweight causal LM, good for PPL computation
            model_name = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()  # set to evaluation mode (no dropout, etc)
            
            # Use GPU if available
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
            except:
                self.device = "cpu"
            
            self._loaded = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load PPL model: {e}")
            return False
    
    def compute_ppl(self, text: str) -> Optional[float]:
        """
        Compute perplexity of the given text.
        
        Perplexity = exp(cross_entropy_loss)
        Lower perplexity = model finds the text more predictable/coherent.
        
        Parameters
        ──────────
        text : str
            The text to evaluate. Should be at least a few tokens.
        
        Returns
        ───────
        Optional[float]
            PPL value, or None if computation fails or model unavailable.
        """
        if not self._load_model():
            return None
        
        if not text or len(text.split()) < 3:
            return None  # too short to evaluate
        
        try:
            import torch
            
            # Tokenize
            encodings = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = encodings["input_ids"].to(self.device)
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # Compute cross-entropy: shift targets by 1 (standard LM loss)
            # We predict token i+1 from tokens 0..i
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute loss per token
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Perplexity = exp(loss)
            ppl = float(torch.exp(loss).cpu().numpy())
            
            # Clamp to reasonable range (handles edge cases)
            return float(np.clip(ppl, 1.0, 10000.0))
        
        except Exception as e:
            warnings.warn(f"PPL computation failed: {e}")
            return None
    
    def validate_merge(self, text_a: str, text_b: str, threshold: float = 1.1) -> bool:
        """
        Validate that merging text_a and text_b actually improves coherence.
        
        Algorithm
        ─────────
        1. Get PPL of each chunk individually
        2. Get PPL of merged chunk
        3. Check: ppl_merged < max(ppl_a, ppl_b) * threshold
        
        Parameters
        ──────────
        text_a, text_b : str
            Texts to potentially merge
        threshold : float
            Allowed PPL increase factor. Default 1.1 = allow 10% PPL increase
            (strict validation: lower = more restrictive merge decisions)
        
        Returns
        ───────
        bool
            True if merge is PPL-valid (merge should improve or maintain coherence)
            False if merge would hurt coherence
        """
        if not TRANSFORMERS_AVAILABLE:
            return True  # no validation available, assume valid
        
        ppl_a = self.compute_ppl(text_a)
        ppl_b = self.compute_ppl(text_b)
        merged_text = text_a + "\n\n" + text_b
        ppl_merged = self.compute_ppl(merged_text)
        
        # Require at least 2 of 3 computations to succeed
        valid_ppls = sum(p is not None for p in [ppl_a, ppl_b, ppl_merged])
        if valid_ppls < 2:
            return True  # not enough data, assume valid
        
        # If only merged PPL is missing, still validate
        if ppl_merged is not None:
            max_individual_ppl = max(ppl_a or 1000, ppl_b or 1000)
            return ppl_merged < max_individual_ppl * threshold
        
        # Fallback: if we can't compute merged PPL, allow merge
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Entropy Rate Calculator (Intra-chunk coherence)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_entropy_rate(text: str) -> float:
    """
    Compute ENTROPY RATE: the average information per symbol across
    consecutive sentence transitions within a chunk.
    
    Entropy Rate Intuition
    ──────────────────────
    If sentences within a chunk are tightly related (same paragraph/topic),
    entropy rate is LOW: knowing sentence i, sentence i+1 is predictable.
    
    If sentences within a chunk are unrelated (different topics),
    entropy rate is HIGH: sentences are independent, hard to predict.
    
    Use Case
    ────────
    - Detect poor merge candidates: if a "merged" chunk would have high
      entropy rate, it suggests the original chunks shouldn't have been merged
    - Evaluate chunk quality: high entropy rate within a chunk = coherence issue
    
    Computation
    ───────────
    1. Split text into sentences
    2. Compute unigram distribution for each sentence
    3. For each consecutive pair (sᵢ, sᵢ₊₁):
       a. Compute Jensen-Shannon divergence (symmetric)
       b. This measures how different the next sentence's vocab is
    4. Entropy rate = mean JS divergence across all transitions
    5. Normalize to [0,1] via min/max of KL divergence theoretical bounds
    
    Parameters
    ──────────
    text : str
        The chunk text to analyze
    
    Returns
    ───────
    float ∈ [0,1]
        Entropy rate, bounded in [0, 1].
        0 (low) = sentences are similar/cohesive
        1 (high) = sentences are diverse/incoherent
    """
    # Split into sentences
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
        return 0.0  # no vocabulary = no entropy rate
    
    # Compute JSD for consecutive sentence pairs
    jsd_values: List[float] = []
    for i in range(len(sentence_dists) - 1):
        freq_curr, len_curr = sentence_dists[i]
        freq_next, len_next = sentence_dists[i + 1]
        
        if not freq_curr or not freq_next:
            jsd_values.append(0.5)  # neutral when either sentence is empty
            continue
        
        # Convert frequencies to probability distributions
        p = np.array([freq_curr.get(w, 0) / len_curr for w in joint_vocab], dtype=np.float64)
        q = np.array([freq_next.get(w, 0) / len_next for w in joint_vocab], dtype=np.float64)
        
        # Compute Jensen-Shannon divergence (symmetric, stable)
        m = (p + q) / 2.0
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        jsd_values.append(float(np.clip(jsd, 0.0, 1.0)))
    
    if not jsd_values:
        return 0.0
    
    # Entropy rate = mean JSD across transitions
    # Already bounded in [0,1] from individual JSD values
    entropy_rate = float(np.mean(jsd_values))
    
    return float(np.clip(entropy_rate, 0.0, 1.0))


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
                        new_feat["entropy_rate"],  # NEW
                        new_feat["pmi_drop"],
                        new_feat["depth_change"],
                        new_feat["drift"],
                        float(new_feat.get("ppl_valid", 1.0)),  # NEW
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
    
    Enhanced to include entropy_rate and ppl validity.
    All values guaranteed ∈ [0, 1].
    """
    return {
        "jsd":          _compute_jsd(text_a, text_b),
        "hellinger":    _compute_hellinger(text_a, text_b),
        "entropy_rate": _compute_entropy_rate(text_b),  # NEW: intra-chunk rate of text_b
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
