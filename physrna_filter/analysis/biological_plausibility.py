"""
Biological plausibility checks for AF3 protein-RNA predictions.

Cross-references predicted binding with:
  - ENCODE eCLIP peaks (Van Nostrand et al. 2020)
  - Basic RBP binding-motif enrichment (ATtRACT-style PWM scoring)

When genomic coordinates or RBP identity are unavailable, returns UNKNOWN
rather than blocking the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Minimal PWM fragments for common RBP families (literature: Ray et al. 2013 ATtRACT)
_MOTIF_PATTERNS: dict[str, list[str]] = {
    "PTB":   [r"CUUC", r"UCUU"],
    "HNRNPA1": [r"UAGGG", r"UAGG"],
    "RBFOX": [r"UGCAUG"],
    "PUMILIO": [r"UGUANAUA", r"UGUA"],
    "PUM1":  [r"UGUA"],
    "MS2":   [r"AACA", r"CAC", r"GGAUCACC"],
    "U1A":   [r"UUUGC", r"AUUGCAC"],
    "HUD":   [r"UUUUAUUUU", r"AUUU"],
    "NOVA":  [r"CAUU", r"ACUU"],
    "NOVA2": [r"CAUU", r"ACUU"],
    "PABP":  [r"A{6,}"],
    "PABPC1": [r"A{6,}"],
    "LIN28": [r"GGAG", r"GGC"],
    "REV":   [r"GGAUC", r"GCGC"],
    "DEFAULT": [r"GGAG", r"CUUC", r"UGUA"],
}


def _poly_a_fraction(seq: str) -> float:
    seq = seq.upper().replace("T", "U")
    if not seq:
        return 0.0
    return seq.count("A") / len(seq)


@dataclass
class BiologicalPlausibilityResult:
  verdict: str = "UNKNOWN"
  eclip_supported: bool | None = None
  motif_hits: list[str] = field(default_factory=list)
  motif_score: float = 0.0
  sequence_match: str = "UNKNOWN"
  sequence_identity: float = float("nan")
  notes: list[str] = field(default_factory=list)


def score_binding_motifs(
    rna_sequence: str,
    rbp_name: str | None = None,
) -> tuple[float, list[str]]:
    """
    Score RNA sequence for known RBP binding motifs.

    Returns (hit_count, list of matched motif names).
    """
    seq = rna_sequence.upper().replace("T", "U")
    key = (rbp_name or "").upper().split()
    patterns = _MOTIF_PATTERNS.get(
        key[0] if key else "",
        _MOTIF_PATTERNS["DEFAULT"],
    )
    if rbp_name:
        for key, pats in _MOTIF_PATTERNS.items():
            if key in rbp_name.upper():
                patterns = pats
                break

    hits = []
    for pat in patterns:
        if re.search(pat, seq):
            hits.append(pat)

    return float(len(hits)), hits


def check_eclip_support(
    rbp_name: str,
    chrom: str,
    start: int,
    end: int,
) -> bool | None:
    """
    Check whether an eCLIP peak overlaps the given genomic interval.

    Returns None if eCLIP data cannot be fetched (network/cache miss).
    """
    try:
        from ..data.fetch_encodeclip import (
            search_eclip_experiments,
            fetch_eclip_peaks,
            check_site_has_eclip_signal,
        )
    except ImportError:
        return None

    try:
        experiments = search_eclip_experiments(rbp_name)
        if not experiments:
            return None

        for exp in experiments[:3]:
            acc = exp.get("accession", "")
            if not acc:
                continue
            peaks = fetch_eclip_peaks(acc)
            if check_site_has_eclip_signal(chrom, start, end, peaks):
                return True
        return False
    except Exception:
        return None


def _pattern_literal_length(pattern: str) -> int:
    """Length of nucleotide literals in a motif regex (ignores anchors/meta)."""
    return len(re.sub(r"[^ACGU]", "", pattern.upper()))


def _specific_motif_hits(hits: list[str], min_len: int = 6) -> list[str]:
    """Motifs long enough to avoid spurious partner-mismatch calls (e.g. CAC)."""
    return [h for h in hits if _pattern_literal_length(h) >= min_len]


def _detect_partner_mismatch(
    rna_sequence: str,
    rbp_name: str,
) -> tuple[bool, str]:
    """
    Return True when RNA sequence motifs match a different RBP better than the
    declared partner (wrong-partner AF3 hallucination).

    Only specific motifs (≥6 nt literals) count — short fragments like MS2 ``CAC``
    appear in unrelated hairpins and must not fail Nova-2 positives.
    """
    declared_score, declared_hits = score_binding_motifs(rna_sequence, rbp_name)
    if _specific_motif_hits(declared_hits):
        return False, ""

    declared_upper = (rbp_name or "").upper()
    best_other = ""
    best_hits: list[str] = []

    for key in _MOTIF_PATTERNS:
        if key == "DEFAULT":
            continue
        if key in declared_upper or declared_upper in key:
            continue
        _, hits = score_binding_motifs(rna_sequence, key)
        strong = _specific_motif_hits(hits)
        if len(strong) > len(best_hits):
            best_other = key
            best_hits = strong

    if best_hits:
        return True, (
            f"RNA matches {best_other} motifs ({', '.join(best_hits)}), "
            f"not {rbp_name}"
        )
    return False, ""


def _normalize_rna(seq: str) -> str:
    return seq.upper().replace("T", "U").strip()


def sequence_match_verdict(
    expected_sequence: str | None,
    observed_sequence: str | None,
    *,
    min_identity: float = 0.85,
) -> tuple[str, float]:
    """
    Compare expected panel/input RNA to the sequence parsed from the structure.

    Wrong-partner AF3 jobs should FAIL when the bound RNA differs from the
    declared input sequence.
    """
    if not expected_sequence or not observed_sequence:
        return "UNKNOWN", float("nan")

    exp = _normalize_rna(expected_sequence)
    obs = _normalize_rna(observed_sequence)
    if exp == obs:
        return "PASS", 1.0

    # Allow trailing/leading unresolved tails if the core matches
    if len(exp) >= 8 and (exp in obs or obs in exp):
        shorter = min(len(exp), len(obs))
        longer = max(len(exp), len(obs))
        identity = shorter / longer
        if identity >= min_identity:
            return "WARN", identity
        return "FAIL", identity

    # Global identity (Levenshtein-free: positional match on min length)
    n = min(len(exp), len(obs))
    if n == 0:
        return "FAIL", 0.0
    matches = sum(1 for i in range(n) if exp[i] == obs[i])
    identity = matches / max(len(exp), len(obs))
    if identity >= min_identity:
        return "WARN", identity
    return "FAIL", identity


def assess_biological_plausibility(
    rna_sequence: str | None = None,
    rbp_name: str | None = None,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    observed_rna_sequence: str | None = None,
    reference_native_sequence: str | None = None,
) -> BiologicalPlausibilityResult:
    """
    Combined biological plausibility assessment.

    Verdict logic:
      - FAIL if eCLIP explicitly contradicts (no peak when coords provided)
        AND no motif support
      - WARN if motif support weak (0 hits) but eCLIP unavailable
      - PASS if eCLIP supports OR ≥1 motif hit
      - UNKNOWN if insufficient metadata
    """
    result = BiologicalPlausibilityResult()

    if reference_native_sequence and rna_sequence:
        ref_verdict, ref_identity = sequence_match_verdict(
            reference_native_sequence,
            rna_sequence,
            min_identity=1.0,
        )
        if ref_verdict != "PASS":
            result.verdict = "FAIL"
            result.sequence_match = ref_verdict
            result.sequence_identity = ref_identity
            result.notes.append(
                "RNA is not the native partner sequence "
                f"(identity={ref_identity:.2f} vs positive control)"
            )
            return result

    if rna_sequence and observed_rna_sequence:
        seq_verdict, identity = sequence_match_verdict(
            rna_sequence, observed_rna_sequence
        )
        result.sequence_match = seq_verdict
        result.sequence_identity = identity
        if seq_verdict == "FAIL":
            result.verdict = "FAIL"
            result.notes.append(
                f"Structure RNA does not match declared input "
                f"(identity={identity:.2f})"
            )
            return result

    if rna_sequence:
        result.motif_score, result.motif_hits = score_binding_motifs(
            rna_sequence, rbp_name
        )
        rbp_upper = (rbp_name or "").upper()
        if _poly_a_fraction(rna_sequence) >= 0.85 and any(
            k in rbp_upper for k in ("PUM", "PUMILIO")
        ):
            result.verdict = "FAIL"
            result.notes.append(
                "Poly-A RNA with Pumilio-family protein — expect UGUA motif, not homopoly-A"
            )
            return result

        if any(k in rbp_upper for k in ("PABP", "PABPC1")):
            if _poly_a_fraction(rna_sequence) < 0.7:
                result.verdict = "FAIL"
                result.notes.append(
                    "Non-poly(A) RNA with PABP — expect homopoly-A tract"
                )
                return result

        if rbp_name:
            mismatch, note = _detect_partner_mismatch(rna_sequence, rbp_name)
            if mismatch:
                result.verdict = "FAIL"
                result.notes.append(note)
                return result

    if rbp_name and chrom and start is not None and end is not None:
        result.eclip_supported = check_eclip_support(rbp_name, chrom, start, end)
        if result.eclip_supported is True:
            result.notes.append(f"eCLIP peak overlaps {chrom}:{start}-{end}")
        elif result.eclip_supported is False:
            result.notes.append(f"No eCLIP peak at {chrom}:{start}-{end}")

    has_motif = result.motif_score > 0
    eclip = result.eclip_supported

    if eclip is True or has_motif:
        result.verdict = "PASS"
    elif eclip is False and not has_motif:
        result.verdict = "FAIL"
        result.notes.append("No eCLIP signal and no binding motif match")
    elif rna_sequence and not has_motif:
        result.verdict = "WARN"
        result.notes.append("No known binding motif detected in RNA sequence")
    else:
        result.verdict = "UNKNOWN"

    return result
