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
from typing import Dict, List, Tuple

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


# ─── LTX Prompt Generator ────────────────────────────────────────────────────

def build_ltx_prompt(
    character_name: str,
    top_action: Dict,
    ssv: SSV,
    nsv: NSV,
    signal: ExternalSignal,
    beat_context: str,
) -> Dict[str, str]:
    action = top_action["action"]

    # ── Camera: קבוע לפי פעולה (לא תלוי SSV) ──
    camera_specs = {
        "accept_invitation":  "Two-shot, eye-level, medium depth, slow push-in toward Maya",
        "flirt_no_commit":    "Two-shot, eye-level, medium depth, conversational rhythm",
        "change_subject":     "Medium shot, slight rack focus, character turns body",
        "confront_married":   "Tight close-up, static frame, very long take, character faces camera",
    }

    # ── Pace: קבוע לפי פעולה ──
    pace_specs = {
        "accept_invitation":  "Medium — drawn toward connection",
        "flirt_no_commit":    "Medium — conversational rhythm",
        "change_subject":     "Quick — escape rhythm",
        "confront_married":   "Very slow — weight in every word",
    }

    # ── Vocal tone: SSV-sensitive ──
    def _vocal_tone(action: str, ssv: SSV) -> str:
        if action == "confront_married":
            if ssv.shame > 0.60:
                return "Raw, slightly breaking — the words cost him"
            elif ssv.shame < 0.25:
                return "Flat, direct, factual — information being stated"
            else:
                return "Controlled, deliberate — firm without collapse"
        elif action == "accept_invitation":
            return "Warm, engaged, quiet — attraction without proclamation"
        elif action == "flirt_no_commit":
            return "Charismatic, slightly louder, rhythmic pauses"
        else:
            return "Flat, controlled, minimal inflection"

    # ── Posture: SSV-sensitive ──
    def _posture(action: str, ssv: SSV) -> str:
        if action == "confront_married":
            if ssv.conflict_avoidance > 0.60:
                return "Contracted inward — body holds the cost, no retreat but no expansion"
            else:
                return "Open, squared toward Maya — direct, no cost absorbed"
        elif action == "accept_invitation":
            return "Leaning in, arms uncrossed, oriented toward Maya"
        elif action == "flirt_no_commit":
            return "Relaxed, confident lean, one arm on bar"
        else:
            return "Rising, squaring with decision, looking for exit"

    # ── Emotion anchor: SSV-sensitive ──
    def _emotion_anchor(action: str, ssv: SSV, signal: ExternalSignal) -> str:
        if not signal.wife_message_active:
            return "Desire / Risk / Excitement"
        if action == "confront_married":
            if ssv.shame > 0.60 and ssv.loyalty > 0.60:
                return "Obligation / Guilt / Relational gravity"
            elif ssv.shame < 0.25:
                return "Clarity / Closure / The fact of the matter"
            else:
                return "Duty / Controlled restraint"
        return "Obligation / Guilt / Relational gravity"

    # ── Performance pause: SSV-sensitive ──
    def _pause_note(action: str, ssv: SSV, nsv: NSV) -> str:
        if action != "confront_married":
            return ""
        pause_ms = int(200 + ssv.shame * nsv.P * 900 + nsv.R * (1 - ssv.RT) * 400)
        if pause_ms > 900:
            return f"\n- Beat: {pause_ms}ms held before words arrive — silence carries the weight"
        elif pause_ms > 500:
            return f"\n- Beat: {pause_ms}ms — brief pause before speaking"
        else:
            return f"\n- Beat: {pause_ms}ms — almost no hesitation"

    # ── Gaze: SSV-sensitive ──
    def _gaze(action: str, ssv: SSV, nsv: NSV) -> str:
        if action == "confront_married":
            gaze_instability = nsv.P * ssv.shame
            if gaze_instability > 0.60:
                return "\n- Gaze: Breaks — avoidance, guilt before action"
            else:
                return "\n- Gaze: Locked and stable — meets her eyes directly"
        return ""

    # ── NSV context ──
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

    compiled_prompt = (
        f"[MoToMo-v4.2 | Deterministic Character Engine]\n"
        f"[Character: {character_name}]\n"
        f"[Beat: Uri at the Bar — Beat 7]\n"
        f"[Location: Bar — Late Evening]\n"
        f"[{beat_context}]\n\n"
        f"ACTION: {top_action['label']}\n\n"
        f"PERFORMANCE DIRECTIVES:\n"
        f"- Posture: {_posture(action, ssv)}\n"
        f"- Voice: {_vocal_tone(action, ssv)}\n"
        f"- Camera: {camera_specs[action]}\n"
        f"- Pace: {pace_specs[action]}"
        f"{_pause_note(action, ssv, nsv)}"
        f"{_gaze(action, ssv, nsv)}\n"
        f"- Emotion anchor: {_emotion_anchor(action, ssv, signal)}\n\n"
        f"INTERNAL STATE: {nsv_context}\n"
        f"MoToMo score={top_action['score']:.3f} "
        f"[loyalty={ssv.loyalty:.2f}, shame={ssv.shame:.2f}, RT={ssv.RT:.2f}] "
        f"computed {action} as most internally consistent given NSV."
    )

    negative_prompt = (
        "No improvisation outside computed action space. "
        "No contradictory emotional signals. "
        "No relaxed or casual demeanor. "
        "No breaking of deterministic character logic."
    )

    return {
        "compiled_prompt": compiled_prompt,
        "negative_prompt": negative_prompt,
        "camera": camera_specs[action],
        "vocal_tone": _vocal_tone(action, ssv),
        "duration": 9 if action != "confront_married" else 12,
        "aspect_ratio": "2.39:1",
        "resolution": "1920x804",
    }
# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize(vec):
    mag = math.sqrt(sum(x**2 for x in vec))
    if mag == 0:
        return vec
    return [x / mag for x in vec]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x * 2))
