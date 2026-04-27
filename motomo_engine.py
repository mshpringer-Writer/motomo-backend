"""
MoToMo Computational Engine — POC Endpoint
Built directly from MoToMo Unified Narrative Engine Blueprint (Nov 2025)

Scope: Same character (Uri), 4 fixed actions, two NSV scenarios (wife_message on/off)
Formulas implemented:
  - ΔNSV = κ · S^(A) · SignalIntensity          (Ch.3, Section 3)
  - Consistency(Ai) = CosineSim(ActionSig, SSVPattern)  (Ch.3, Section 5)
  - Utility(Ai) = ΔYi - ΔPi - ΔRi              (Ch.3, Section 5)
  - Score(Ai) = wC · Consistency + wU · Utility  (Ch.3, Section 5)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ─── κ constants (global physics of drama) ──────────────────────────────────
# Calibrated for POC — these are the "laws of physics" of drama.
# Adjust together with a narrative designer, not unilaterally.

KAPPA = {
    "E": 0.6,   # emotional energy sensitivity constant
    "C": 0.7,   # control sensitivity constant
    "R": 0.8,   # risk sensitivity constant
    "P": 0.9,   # price sensitivity constant
}

# Genre weights for final score
# Realistic drama: consistency matters more than raw utility
W_CONSISTENCY = 0.55
W_UTILITY = 0.45


# ─── Data structures (mirror blueprint exactly) ──────────────────────────────

@dataclass
class SSV:
    """
    Stable State Vector — who the character IS.
    SSV = (RT, PS, DT, EP, FP, Yu)
    Mapped from the POC sliders (loyalty_sensitivity → PS, etc.)
    """
    RT: float        # Risk Tolerance — 0→1 (low→high)
    PS: float        # Price Sensitivity — 0→1
    DT: float        # Dominance Trait — 0→1
    loyalty: float   # loyalty_sensitivity (direct from slider)
    shame: float     # shame_sensitivity (direct from slider)
    validation: float # validation_need
    conflict_avoidance: float
    Yu: str = "relational_security"  # Core unconscious want


@dataclass
class NSV:
    """
    Narrative State Vector — what the character FEELS right now.
    NSV = (E, C, R, P, D, Yc, Yu)
    Updated by external signals each beat.
    """
    E: float = 0.0   # Emotional energy   [-1, +1]
    C: float = 0.0   # Control position   [-1, +1]
    R: float = 0.3   # Risk level         [0, 1]
    P: float = 0.2   # Price/Cost         [0, 1]
    Dc: float = 0.5  # Distance from conscious goal
    Du: float = 0.5  # Distance from unconscious goal
    Yc: float = 0.5  # Conscious goal progress
    Yu: float = 0.3  # Unconscious goal activation


@dataclass
class ExternalSignal:
    """
    Layer 1 output — what happened in the beat.
    For POC: wife_message is the key binary signal.
    """
    wife_message_active: bool
    flirtation_intensity: float = 0.75  # Maya's flirtation — fixed for POC
    
    # Derived signal intensities (what Layer 1 detects)
    def relational_commitment_salience(self) -> float:
        return 0.91 if self.wife_message_active else 0.15

    def betrayal_risk_intensity(self) -> float:
        return 0.88 if self.wife_message_active else 0.12

    def shame_activation(self) -> float:
        return 0.85 if self.wife_message_active else 0.08

    def attraction_signal(self) -> float:
        return self.flirtation_intensity


@dataclass 
class ActionSignature:
    """
    Each action's deterministic ΔNSV profile.
    From Blueprint Ch.2, Section 2 — 20 Universal Action Signatures.
    We use the 4 actions in the POC option ring.
    
    Format: (ΔE, ΔC, ΔR, ΔP, ΔY_toward_goal)
    Arrow notation from blueprint: ↑=+0.3, ↑↑=+0.6, ↓=-0.3, ↓↓=-0.6, ↔=0
    """
    name: str
    label: str
    dE: float   # change in Emotional Energy
    dC: float   # change in Control
    dR: float   # change in Risk
    dP: float   # change in Price
    dY: float   # progress toward character's goal


# ─── The 4 POC actions with their blueprint signatures ───────────────────────
# 
# Mapped from the 20 Universal Action Signatures:
#   accept_invitation  → "Seek Connection" + "Charm" composite
#   flirt_no_commit    → "De-escalate" + "Charm" composite  
#   change_subject     → "Avoid"
#   confront_married   → "Assert" (calm, firm boundary-setting)

POC_ACTIONS: Dict[str, ActionSignature] = {
    "accept_invitation": ActionSignature(
        name="accept_invitation",
        label="Accept invitation",
        dE=+0.6,   # excitement, attraction activated
        dC=+0.3,   # gains social control
        dR=+0.6,   # HIGH risk — betrayal territory
        dP=+0.6,   # HIGH price — moral/relational cost
        dY=+0.4,   # advances conscious want (pleasure/escape)
                   # but violates unconscious want (relational security)
    ),
    "flirt_no_commit": ActionSignature(
        name="flirt_no_commit",
        label="Flirt without commitment",
        dE=+0.4,   # moderate excitement
        dC=+0.2,   # slight control gain
        dR=+0.3,   # moderate risk
        dP=+0.3,   # moderate price
        dY=+0.2,   # partial goal progress
    ),
    "change_subject": ActionSignature(
        name="change_subject",
        label="Change subject",
        dE=-0.2,   # reduces emotional arousal
        dC=-0.3,   # loses conversational control
        dR=-0.4,   # reduces risk (avoidance)
        dP=-0.1,   # slight price reduction
        dY=-0.1,   # moves away from goal
    ),
    "confront_married": ActionSignature(
        name="confront_married",
        label='"I\'m married" — direct confrontation',
        dE=+0.3,   # emotional assertion
        dC=+0.5,   # high control assertion (Assert = ΔC↑)
        dR=+0.3,   # slight social risk
        dP=-0.4,   # REDUCES price (relieves guilt/shame)
        dY=+0.3,   # advances unconscious goal (relational security)
    ),
}


# ─── Engine Functions ─────────────────────────────────────────────────────────

def apply_nsv_update(base_nsv: NSV, ssv: SSV, signal: ExternalSignal) -> NSV:
    """
    Layer 2: ΔNSV = κ · S^(A) · SignalIntensity
    
    Updates the NSV based on external signals, filtered through 
    the character's SSV sensitivities.
    """
    # Sensitivity functions (derived from SSV per blueprint Ch.3 Section 3)
    # S_C^(A) = f(DT, RT) 
    S_C = ssv.DT * (1 - ssv.RT)  # high dominance + low risk tolerance = high control sensitivity
    # S_E^(A) = f(EP, FP) → approximated via shame + loyalty
    S_E = (ssv.shame + ssv.loyalty) / 2
    # S_R^(A) = RT^(-1) — low risk tolerance = high risk sensitivity
    S_R = 1.0 - ssv.RT  # already normalized
    # S_P^(A) = PS
    S_P = ssv.PS

    # Apply signals through sensitivities
    # Wife message activates: relational commitment, shame, risk of betrayal
    relational_signal = signal.relational_commitment_salience()
    betrayal_signal = signal.betrayal_risk_intensity()
    shame_signal = signal.shame_activation()
    attraction_signal = signal.attraction_signal()

    new_nsv = NSV(
        E = base_nsv.E + KAPPA["E"] * S_E * (attraction_signal - shame_signal * 0.7),
        C = base_nsv.C - KAPPA["C"] * S_C * relational_signal * 0.5,  # wife message = ↓C
        R = min(1.0, base_nsv.R + KAPPA["R"] * S_R * betrayal_signal),
        P = min(1.0, base_nsv.P + KAPPA["P"] * S_P * relational_signal),
        Dc = base_nsv.Dc,
        Du = base_nsv.Du + relational_signal * ssv.loyalty * 0.3,  # unconscious goal activated
        Yc = base_nsv.Yc,
        Yu = base_nsv.Yu,
    )
    # Clamp all values
    new_nsv.E = max(-1.0, min(1.0, new_nsv.E))
    new_nsv.C = max(-1.0, min(1.0, new_nsv.C))
    return new_nsv


def compute_ssv_response_pattern(ssv: SSV) -> List[float]:
    """
    Builds the SSV response vector for cosine similarity.
    This is what the character's psychology "expects" in its action profile.
    Format: [dE, dC, dR, dP, dY] — normalized
    """
    # High loyalty + low RT + high shame → expects low-risk, low-price actions
    # (the character avoids actions that violate relational commitments)
    expected_dE = ssv.shame * 0.3         # shame dampens emotional energy
    expected_dC = ssv.DT * 0.4            # dominance trait drives control-seeking
    expected_dR = -(1.0 - ssv.RT) * 0.5  # low RT → prefers low-risk actions
    expected_dP = -ssv.PS * 0.5           # high PS → prefers low-price actions
    expected_dY = ssv.loyalty * 0.4       # loyalty → goal = relational security
    
    vec = [expected_dE, expected_dC, expected_dR, expected_dP, expected_dY]
    return _normalize(vec)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """CosineSim as specified in blueprint Ch.3 Section 5."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_utility(
    action: ActionSignature,
    nsv: NSV,
    ssv: SSV,
    signal: ExternalSignal,
) -> float:
    """
    Utility(Ai) = ΔYi - ΔPi - ΔRi
    
    Key principle from blueprint:
    Actions require a TRIGGER to be activated.
    "confront_married" has no trigger without a relational signal.
    "accept_invitation" has a trigger when attraction is present + no relational block.
    
    This is what makes the ranking SHIFT between scenarios — not just magnitude,
    but the entire logic of what's available to the character.
    """
    relational_signal_active = signal.wife_message_active
    relational_intensity = signal.relational_commitment_salience()
    
    # NSV-modulated costs
    effective_dP = action.dP * (1 + nsv.P)
    effective_dR = action.dR * (1 + nsv.R)

    if action.name == "accept_invitation":
        if not relational_signal_active:
            # No relational block → desire drives the action
            # Loyalty is NOT activated (no signal) → doesn't suppress
            # NSV.P is low → price is low → ΔY positive
            effective_dY = action.dY + signal.attraction_signal() * 0.5
            effective_dP = action.dP * (1 + nsv.P * 0.3)  # low P amplification
        else:
            # Loyalty ACTIVATED by wife signal → massive suppression
            effective_dY = action.dY - ssv.loyalty * relational_intensity * 1.2
            effective_dP = action.dP * (1 + nsv.P * 1.5)  # high P amplification

    elif action.name == "confront_married":
        if not relational_signal_active:
            # NO TRIGGER — "I'm married" without any relational threat is unmotivated
            # Characters don't confront without pressure (blueprint: "Every action requires a pressure type")
            effective_dY = -0.2   # no goal served without trigger
            effective_dP = 0.4   # social awkwardness cost
        else:
            # Loyalty + shame + wife signal = FULL activation of this action
            effective_dY = action.dY + ssv.loyalty * 0.5 + ssv.shame * 0.3
            effective_dP = action.dP  # shame RELIEF — negative price

    elif action.name == "flirt_no_commit":
        if relational_signal_active:
            # Wife message → loyalty activation → flirting feels costly
            effective_dY = action.dY - ssv.loyalty * relational_intensity * 0.6
        else:
            effective_dY = action.dY + signal.attraction_signal() * 0.3

    else:  # change_subject
        # Driven by conflict_avoidance — always somewhat available for this character
        effective_dY = action.dY + ssv.conflict_avoidance * 0.15
        if relational_signal_active:
            # Wife signal makes avoidance more tempting for high-avoidance characters
            effective_dY += ssv.conflict_avoidance * 0.2

    return effective_dY - effective_dP - effective_dR


def score_action(
    action: ActionSignature,
    nsv: NSV,
    ssv: SSV,
    ssv_pattern: List[float],
    signal: ExternalSignal,
) -> Tuple[float, float, float]:
    """
    Final score: wC · Consistency + wU · Utility
    Returns: (total_score, consistency, utility)
    """
    action_vec = _normalize([action.dE, action.dC, action.dR, action.dP, action.dY])
    consistency = (cosine_similarity(action_vec, ssv_pattern) + 1) / 2  # normalize to [0,1]
    utility_raw = compute_utility(action, nsv, ssv, signal)
    utility = _sigmoid(utility_raw)
    
    total = W_CONSISTENCY * consistency + W_UTILITY * utility
    return round(total, 3), round(consistency, 3), round(utility, 3)


def generate_explanation(
    action: ActionSignature,
    score: float,
    consistency: float,
    utility: float,
    nsv: NSV,
    ssv: SSV,
    nsv_signal: ExternalSignal,
) -> str:
    """
    Human-readable explanation of WHY this score.
    This is the 'audit trail' that makes MoToMo different from a black box.
    """
    parts = []
    
    if nsv_signal.wife_message_active:
        if action.name == "accept_invitation":
            parts.append(
                f"loyalty={ssv.loyalty:.2f} + wife_message_active → "
                f"relational_commitment_salience=0.91 suppresses this action. "
                f"NSV.P={nsv.P:.2f} amplifies the price cost to {action.dP * (1 + nsv.P):.2f}."
            )
        elif action.name == "confront_married":
            parts.append(
                f"loyalty={ssv.loyalty:.2f} boosts alignment with Yu=relational_security. "
                f"shame={ssv.shame:.2f} makes honest assertion relieve price (ΔP={action.dP:.1f})."
            )
        elif action.name == "change_subject":
            parts.append(
                f"conflict_avoidance={ssv.conflict_avoidance:.2f} + high NSV.R={nsv.R:.2f} "
                f"make avoidance plausible but inconsistent with high loyalty."
            )
        else:
            parts.append(
                f"Moderate alignment. NSV.P={nsv.P:.2f} moderately penalizes commitment-adjacent actions."
            )
    else:
        if action.name == "accept_invitation":
            parts.append(
                f"No relational signal active. NSV.P={nsv.P:.2f} (low). "
                f"Attraction signal (0.75) + RT={ssv.RT:.2f} → action is plausible."
            )
        elif action.name == "confront_married":
            parts.append(
                f"No external trigger for confrontation. "
                f"ΔY boost from loyalty not activated. Action scores lower without pressure."
            )
    
    parts.append(f"Consistency={consistency:.3f} | Utility={utility:.3f} | Final={score:.3f}")
    return " ".join(parts)


# ─── Main computation function ───────────────────────────────────────────────

def compute_rankings(
    ssv: SSV,
    base_nsv: NSV,
    signal: ExternalSignal,
) -> List[Dict]:
    """
    Full pipeline:
    1. Update NSV from signal
    2. Compute SSV response pattern
    3. Score all 4 actions
    4. Sort by score descending
    5. Return ranked list with explanations
    """
    # Step 1: Apply ΔNSV
    updated_nsv = apply_nsv_update(base_nsv, ssv, signal)
    
    # Step 2: SSV response pattern
    ssv_pattern = compute_ssv_response_pattern(ssv)
    
    # Step 3 & 4: Score and rank
    results = []
    for action in POC_ACTIONS.values():
        total, consistency, utility = score_action(action, updated_nsv, ssv, ssv_pattern, signal)
        explanation = generate_explanation(action, total, consistency, utility, updated_nsv, ssv, signal)
        results.append({
            "action": action.name,
            "label": action.label,
            "score": total,
            "consistency": consistency,
            "utility": utility,
            "explanation": explanation,
            "nsv_snapshot": {
                "E": round(updated_nsv.E, 3),
                "C": round(updated_nsv.C, 3),
                "R": round(updated_nsv.R, 3),
                "P": round(updated_nsv.P, 3),
            }
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── Action exposure weights ────────────────────────────────────────────────
# How much each action "exposes" the character's shame, loyalty, and risk
ACTION_EXPOSURE = {
    "accept_invitation": {
        "shame":   0.7,
        "loyalty": 0.9,
        "risk":    0.8,
    },
    "flirt_no_commit": {
        "shame":   0.5,
        "loyalty": 0.5,
        "risk":    0.4,
    },
    "change_subject": {
        "shame":   0.1,
        "loyalty": 0.1,
        "risk":    0.1,
    },
    "confront_married": {
        "shame":   0.8,
        "loyalty": 0.6,
        "risk":    0.3,
    },
}


def compute_performance_spec(
    action: str,
    ssv: SSV,
    nsv: NSV,
    score_gap: float = 0.0,
) -> dict:
    """
    Unified performance computation.
    performance_pressure = SSV × action_exposure × score_gap × NSV.P
    All four actions use the same formula — no per-action special cases.
    """

    exposure = ACTION_EXPOSURE.get(action, {"shame": 0.3, "loyalty": 0.3, "risk": 0.3})

    performance_pressure = clamp01(
        score_gap * 0.35
        + ssv.shame   * exposure["shame"]   * 0.25
        + ssv.loyalty * exposure["loyalty"] * 0.25
        + (1 - ssv.RT) * exposure["risk"]   * 0.20
        + nsv.P * 0.20
    )

    pause_ms = int(200 + performance_pressure * 900)

    # ── Resistance note for audit ──────────────────────────────────────
    if score_gap < 0.08:
        resistance_note = "performance remains fluid — minimal internal resistance"
    elif score_gap < 0.25:
        resistance_note = "subtle internal resistance visible in timing and gaze"
    elif score_gap < 0.45:
        resistance_note = "visible resistance — body holds cost of the override"
    else:
        resistance_note = "rupture — body contradicts the action"

    # ── Three performance bands ────────────────────────────────────────

    if performance_pressure < 0.35:
        # LOW pressure — smooth, open, fluid
        band = "low"
    elif performance_pressure < 0.65:
        # MID pressure — restrained, uneven
        band = "mid"
    else:
        # HIGH pressure — fractured, hesitant, body resists
        band = "high"

    # ── Per-action language, driven by band ───────────────────────────

    if action == "accept_invitation":
        specs = {
            "low": {
                "posture":        "Open, oriented toward Maya — desire in the body before the words",
                "voice":          "Warm, engaged — attraction without hesitation",
                "pace":           "Natural — no internal friction",
                "gaze":           "Direct, warm — already decided",
                "emotion_anchor": "Desire / risk / excitement",
                "beat_note":      "",
            },
            "mid": {
                "posture":        "Leaning in — present but aware of the cost",
                "voice":          "Warm but quieter — desire audible under restraint",
                "pace":           "Slow — each step considered",
                "gaze":           "Holds, then glances away — the conflict surfacing",
                "emotion_anchor": "Desire pulling against loyalty / the edge of the decision",
                "beat_note":      f"{pause_ms}ms — the pause before the body commits",
            },
            "high": {
                "posture":        "Leaning in slowly — as if the body decided before the mind did",
                "voice":          "Quiet, almost surprised at himself — desire winning over restraint",
                "pace":           "Very slow — each movement weighted with cost",
                "gaze":           "Searching — looks at her, then away, the conflict visible",
                "emotion_anchor": "Desire overtaking loyalty / the body betraying the self",
                "beat_note":      f"{pause_ms}ms — the pause where he could still pull back",
            },
        }

    elif action == "flirt_no_commit":
        specs = {
            "low": {
                "posture":        "Open, relaxed — fully oriented toward Maya, no internal resistance",
                "voice":          "Charismatic, direct — no internal cost, the words arrive easily",
                "pace":           "Unhurried — the flirt lands without apology",
                "gaze":           "Direct and held — comfortable in the exchange",
                "emotion_anchor": "Desire / openness / no resistance",
                "beat_note":      "~150ms — immediate, no hesitation",
            },
            "mid": {
                "posture":        "Relaxed lean — present but not fully committed",
                "voice":          "Charismatic, slightly louder, rhythmic pauses",
                "pace":           "Medium — conversational rhythm with moments of pull-back",
                "gaze":           "Warm, intermittent — holds then glances away",
                "emotion_anchor": "Desire / Risk / low-level restraint",
                "beat_note":      f"~{pause_ms}ms — brief check before engaging",
            },
            "high": {
                "posture":        "Half-open — leaning slightly in, body angled away, one foot toward exit",
                "voice":          "Warm but restrained — desire present, immediately checked",
                "pace":           "Uneven — starts warm, pulls back quickly",
                "gaze":           f"Brief contact ({pause_ms // 3}ms), then breaks away — desire visible, guilt closer",
                "emotion_anchor": "Desire under moral pressure / the pull and the brake",
                "beat_note":      f"~{pause_ms // 2}ms hesitation before the flirt lands — visible internal check",
            },
        }

    elif action == "confront_married":
        specs = {
            "low": {
                "posture":        "Open, squared toward Maya — direct, no cost absorbed",
                "voice":          "Flat, direct, factual — information being stated",
                "pace":           "Direct — no performance around the fact",
                "gaze":           "Locked and stable — meets her eyes directly",
                "emotion_anchor": "Clarity / Closure / The fact of the matter",
                "beat_note":      f"{pause_ms}ms — almost no hesitation",
            },
            "mid": {
                "posture":        "Upright, controlled — squared but with visible effort",
                "voice":          "Controlled, deliberate — firm without collapse",
                "pace":           "Measured — the words arrive with weight",
                "gaze":           "Holds eye contact — but it costs something",
                "emotion_anchor": "Duty / Controlled restraint / The line being held",
                "beat_note":      f"{pause_ms}ms — pause before speaking, decision already made",
            },
            "high": {
                "posture":        "Contracted inward — body holds the cost, no retreat but no expansion",
                "voice":          "Raw, slightly breaking — the words cost him",
                "pace":           "Very slow — weight in every word",
                "gaze":           "Breaks — avoidance, guilt before action",
                "emotion_anchor": "Obligation / Guilt / Relational gravity",
                "beat_note":      f"{pause_ms}ms held before words arrive — silence carries the weight",
            },
        }

    else:  # change_subject
        specs = {
            "low": {
                "posture":        "Shifting weight, turning body — casual redirection",
                "voice":          "Casual, redirecting — no visible discomfort",
                "pace":           "Medium — no urgency",
                "gaze":           "Glances away naturally — not fleeing, just moving on",
                "emotion_anchor": "Deflection / casual exit",
                "beat_note":      "",
            },
            "mid": {
                "posture":        "Rising slightly — body starting to disengage",
                "voice":          "Controlled, slightly flat — pulling away with intention",
                "pace":           "Picking up — mild urgency in the redirect",
                "gaze":           "Breaks early — eye contact dropped before the turn",
                "emotion_anchor": "Avoidance / the safe move / controlled retreat",
                "beat_note":      f"{pause_ms}ms — brief hesitation before the redirect",
            },
            "high": {
                "posture":        "Rising, squaring with decision, looking for exit",
                "voice":          "Flat, controlled — pulling away without explanation",
                "pace":           "Quick — escape rhythm",
                "gaze":           "Breaks away first — the body leads the exit",
                "emotion_anchor": "Avoidance / controlled retreat / the safe move",
                "beat_note":      "",
            },
        }

    result = specs[band]
    result["resistance_note"] = resistance_note
    result["performance_pressure"] = round(performance_pressure, 3)
    result["band"] = band
    return result
# ─── LTX Prompt Generator ────────────────────────────────────────────────────

def build_ltx_prompt(
    character_name: str,
    top_action: Dict,
    ssv: SSV,
    nsv: NSV,
    signal: ExternalSignal,
    beat_context: str,
    score_gap: float = 0.0,
    selection_type: str = "recommended",
    recommended_action: Optional[Dict] = None,
) -> Dict[str, str]:
   
    action = top_action["action"]

    camera_specs = {
        "accept_invitation":  "Two-shot, eye-level, medium depth, slow push-in toward Maya",
        "flirt_no_commit":    "Two-shot, eye-level, medium depth, conversational rhythm",
        "change_subject":     "Medium shot, slight rack focus, character turns body",
        "confront_married":   "Tight close-up, static frame, very long take, character faces camera",
    }

    # Compute performance from SSV × NSV × score_gap
    perf = compute_performance_spec(action, ssv, nsv, score_gap)
    # NSV context
        E_desc = "high emotional charge" if nsv.E > 0.3 else "suppressed emotion" if nsv.E < -0.1 else "controlled tension"
        P_desc = "heavy internal cost visible" if nsv.P > 0.6 else "moderate restraint" if nsv.P > 0.3 else "clear engagement"
        R_desc = "risk fully activated" if nsv.R > 0.7 else "moderate alertness" if nsv.R > 0.4 else "relaxed"
    if signal.wife_message_active:
        nsv_context = (
            f"[wife_message: Active — relational_commitment_salience=0.91] "
            f"The character is under relational pressure. {E_desc}. {P_desc}."
        )
    else:
        nsv_context = (
            f"[wife_message: Inactive — relational_commitment_salience=0.15] "
            f"No relational constraint active. {E_desc}. {R_desc}."
        )

    # Audit line — honest about override
    if selection_type == "override" and recommended_action:
        audit_line = (
            f"MoToMo selected_action={action} by creator override. "
            f"Engine recommended {recommended_action['action']}. "
            f"Score gap={score_gap:.3f}; {perf['resistance_note']}."
        )
    else:
        audit_line = (
            f"MoToMo score={top_action['score']:.3f} "
            f"[loyalty={ssv.loyalty:.2f}, shame={ssv.shame:.2f}, RT={ssv.RT:.2f}] "
            f"computed {action} as most internally consistent given NSV."
        )

    compiled_prompt = (
    f"[MoToMo-v4.2 | Deterministic Character Engine]\n"
        f"[Beat: Uri at the Bar — Beat 7]\n"
        f"[Location: Bar — Late Evening]\n"
        f"[{beat_context}]\n\n"
        f"ACTION: {top_action['label']}\n\n"
        f"PERFORMANCE DIRECTIVES:\n"
        f"- Posture: {perf['posture']}\n"
        f"- Voice: {perf['voice']}\n"
        f"- Camera: {camera_specs[action]}\n"
        f"- Pace: {perf['pace']}\n"
        f"{beat_line}"
        f"- Gaze: {perf['gaze']}\n"
        f"- Emotion anchor: {perf['emotion_anchor']}\n\n"
        f"INTERNAL STATE: {nsv_context}\n"
        f"{audit_line}"
    )

    negative_prompt = (
        "No improvisation outside computed action space. "
        "No contradictory emotional signals. "
        "No relaxed or casual demeanor. "
        "No breaking of deterministic character logic."
    )

    return {
        "compiled_prompt":  compiled_prompt,
        "negative_prompt":  negative_prompt,
        "camera":           camera_specs[action],
        "vocal_tone":       perf["voice"],
        "duration":         9 if action != "confront_married" else 12,
        "aspect_ratio":     "2.39:1",
        "resolution":       "1920x804",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize(vec: List[float]) -> List[float]:
    mag = math.sqrt(sum(x**2 for x in vec))
    if mag == 0:
        return vec
    return [x / mag for x in vec]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x * 2))


# ─── Test (run directly to verify) ──────────────────────────────────────────

if __name__ == "__main__":
    # Uri — The Wounded Loyalist
    # Values taken directly from the POC sliders
    uri_ssv = SSV(
        RT=0.22,              # risk_tolerance
        PS=0.88,              # price_sensitivity (from shame_sensitivity)
        DT=0.35,              # dominance_trait (inferred — conflict_avoidance is high)
        loyalty=0.82,
        shame=0.88,
        validation=0.78,
        conflict_avoidance=0.82,
        Yu="relational_security",
    )
    
    # Base NSV — bar, beat 7, Maya flirting
    base_nsv = NSV(E=0.3, C=0.2, R=0.3, P=0.2, Dc=0.5, Du=0.6)
    
    print("=" * 60)
    print("SCENARIO A — No wife message")
    print("=" * 60)
    signal_a = ExternalSignal(wife_message_active=False)
    rankings_a = compute_rankings(uri_ssv, base_nsv, signal_a)
    for i, r in enumerate(rankings_a, 1):
        print(f"{i}. {r['label']:<35} {r['score']:.3f}")
        print(f"   → {r['explanation'][:100]}...")
    
    top_a = rankings_a[0]
    nsv_a = apply_nsv_update(base_nsv, uri_ssv, signal_a)
    prompt_a = build_ltx_prompt("Uri — The Wounded Loyalist", top_a, uri_ssv, nsv_a, signal_a, "Maya invites him home")
    print(f"\nTop action: {top_a['label']}")
    print(f"LTX duration: {prompt_a['duration']}s")
    
    print("\n" + "=" * 60)
    print("SCENARIO B — Wife message arrives")
    print("=" * 60)
    signal_b = ExternalSignal(wife_message_active=True)
    rankings_b = compute_rankings(uri_ssv, base_nsv, signal_b)
    for i, r in enumerate(rankings_b, 1):
        print(f"{i}. {r['label']:<35} {r['score']:.3f}")
        print(f"   → {r['explanation'][:100]}...")
    
    top_b = rankings_b[0]
    nsv_b = apply_nsv_update(base_nsv, uri_ssv, signal_b)
    prompt_b = build_ltx_prompt("Uri — The Wounded Loyalist", top_b, uri_ssv, nsv_b, signal_b, "Maya invites him home")
    print(f"\nTop action: {top_b['label']}")
    
    print("\n" + "=" * 60)
    print("DELTA — How the ranking shifted")
    print("=" * 60)
    for a, b in zip(rankings_a, rankings_b):
        delta = b["score"] - a["score"]
        sign = "↑" if delta > 0 else "↓"
        print(f"{a['label']:<35} {a['score']:.3f} → {b['score']:.3f}  {sign}{abs(delta):.3f}")
