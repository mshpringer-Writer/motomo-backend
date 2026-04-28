"""
MoToMo Behavioral Translation Layer — v0.3.1
Spec: MoToMo_Behavioral_Translation_Layer___Spec_v01_280426.pdf

Changelog from v0.2:
    [math]         price_sensitivity = 1 - SSV.P  (P is tolerance, not sensitivity)
    [math]         inhibition now action-sensitive via world_shame × shame_exposure
                   and bond_conflict × loyalty_conflict
    [math]         violation_pressure added as driver (exported in debug)
    [arch]         pressure_band computed once, passed into action compilers
    [render]       OMIT_NONE_CHANNELS — object_interaction=none silently omitted,
                   smile_state=none keeps "No smile appears."
    [render]       distance_behavior moved earlier in RENDER_ORDER (spatial before kinetic)
    [compiler]     accept_invitation gains MID band case
    [compiler]     change_subject HIGH → distance_behavior=retreat (not holds_distance)
    [safety]       get_nested safe extraction retained from v0.2-A
    [safety]       strict mode + production fallback retained from v0.2-A
    [debug]        implemented_action flag retained from v0.2-A
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Optional

COMPILER_VERSION = "behavioral_compiler_v0.3.1"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _f(d: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Safe float getter with type coercion."""
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def get_nested(payload: Mapping[str, Any], key: str,
               default: Optional[Dict] = None) -> Dict:
    """Safe nested dict extraction with type guard."""
    value = payload.get(key, default if default is not None else {})
    if not isinstance(value, dict):
        return default if default is not None else {}
    return value


# ---------------------------------------------------------------------------
# Action metadata
# ---------------------------------------------------------------------------

ACTION_META: Dict[str, Dict[str, float]] = {
    "flirt_no_commit": {
        "desire_indulgence": 0.65,
        "risk_exposure":     0.45,
        "shame_exposure":    0.50,
        "loyalty_conflict":  0.55,
        "world_violation":   0.45,
    },
    "change_subject": {
        "desire_indulgence": 0.10,
        "risk_exposure":     0.10,
        "shame_exposure":    0.10,
        "loyalty_conflict":  0.10,
        "world_violation":   0.05,
    },
    "accept_invitation": {
        "desire_indulgence": 0.90,
        "risk_exposure":     0.80,
        "shame_exposure":    0.70,
        "loyalty_conflict":  0.90,
        "world_violation":   0.80,
    },
    "confront_married": {
        "desire_indulgence": 0.20,
        "risk_exposure":     0.30,
        "shame_exposure":    0.80,
        "loyalty_conflict":  0.60,
        # negative: confronting is boundary-preserving, not a violation.
        # abs() is applied in driver computation.
        "world_violation":   0.00,   # boundary-asserting, not a violation
    },
    "withdraw": {
        "desire_indulgence": 0.05,
        "risk_exposure":     0.05,
        "shame_exposure":    0.20,
        "loyalty_conflict":  0.15,
        "world_violation":   0.00,
    },
}

# Per-action HIGH threshold — moral stakes differ by action type.
HIGH_THRESHOLDS: Dict[str, float] = {
    "flirt_no_commit":   0.55,
    "change_subject":    0.55,
    "accept_invitation": 0.50,
    "confront_married":  0.60,
    "withdraw":          0.60,
}

MID_THRESHOLD = 0.30

# Intersection: only actions that have BOTH metadata AND a compiler.
# Prevents a future ACTION_META entry without a compiler from appearing implemented.


# ---------------------------------------------------------------------------
# Driver computation
# ---------------------------------------------------------------------------

def compute_behavioral_drivers(
    ssv: Mapping[str, Any],
    nsv: Mapping[str, Any],
    world: Mapping[str, Any],
    action: Mapping[str, Any],
    score_gap: float,
) -> Dict[str, float]:
    """
    Translate SSV × NSV × World × action_meta → intermediate behavioral drivers.

    Key semantics:
        SSV.P = price *tolerance*  →  price_sensitivity = 1 - SSV.P
        inhibition is action-sensitive: a high-shame character doing change_subject
        has lower inhibition than the same character doing accept_invitation.
        world_shame and bond_conflict encode that specificity.
    """
    # SSV
    ssv_R = clamp01(_f(ssv, "R", 0.50))   # risk orientation
    ssv_W = clamp01(_f(ssv, "W", 0.50))   # world-code sensitivity
    ssv_C = clamp01(_f(ssv, "C", 0.50))   # control / agency need
    ssv_P = clamp01(_f(ssv, "P", 0.50))   # price tolerance
    ssv_D = clamp01(_f(ssv, "D", 0.50))   # desire / need pressure
    ssv_A = clamp01(_f(ssv, "A", 0.50))   # attachment / relational binding

    # NSV
    nsv_E     = clamp01(_f(nsv, "E",     0.50))
    nsv_C     = clamp01(_f(nsv, "C",     0.50))
    nsv_P_imm = clamp01(_f(nsv, "P_imm", _f(nsv, "P", 0.0)))
    nsv_P_lat = clamp01(_f(nsv, "P_lat", 0.0))
    nsv_W     = clamp01(_f(nsv, "W",     0.50))

    # World
    world_M = clamp01(_f(world, "M", 0.50))   # moral strictness
    world_S = clamp01(_f(world, "S", 0.50))   # social surveillance
    world_B = clamp01(_f(world, "B", 0.50))   # bond sacredness

    # Action
    desire_indulgence = clamp01(_f(action, "desire_indulgence", 0.0))
    risk_exposure     = clamp01(_f(action, "risk_exposure",     0.0))
    shame_exposure    = clamp01(_f(action, "shame_exposure",    0.0))
    loyalty_conflict  = clamp01(_f(action, "loyalty_conflict",  0.0))
    world_violation   = clamp01(abs(_f(action, "world_violation", 0.0)))

    score_gap = clamp01(score_gap)

    # Derived sensitivity (price tolerance inverted)
    price_sensitivity = clamp01(1.0 - ssv_P)

    # Intermediate shame / bond / violation — action-sensitive
    world_shame       = clamp01(ssv_W * world_M * shame_exposure)
    bond_conflict     = clamp01(ssv_A * world_B * loyalty_conflict)
    violation_pressure = clamp01(
        (world_shame + bond_conflict + world_violation * nsv_W) / 3.0
    )

    # Core drivers
    attraction = clamp01(
        nsv_E             * 0.35
        + ssv_D           * 0.35
        + desire_indulgence * 0.30
    )

    inhibition = clamp01(
        world_shame                  * 0.25
        + bond_conflict              * 0.25
        + nsv_P_imm * price_sensitivity * 0.20
        + nsv_P_lat * price_sensitivity * 0.15
        + score_gap                  * 0.15
    )

    pressure = clamp01(
        score_gap                           * 0.35
        + nsv_P_imm * price_sensitivity     * 0.20
        + nsv_W                             * 0.15
        + inhibition                        * 0.20
        + (1.0 - ssv_R) * risk_exposure     * 0.10
        + violation_pressure                * 0.15
    )

    control_strain = clamp01(abs(ssv_C - nsv_C))

    exposure = clamp01(
        nsv_P_imm * 0.40
        + nsv_W   * 0.30
        + world_S * 0.30
    )

    approach_drive = clamp01(
        attraction   * 0.65
        + ssv_D      * 0.20
        - inhibition * 0.25
    )

    conflict_load = clamp01(
        abs(attraction - inhibition) * 0.35
        + pressure                   * 0.35
        + score_gap                  * 0.20
        + violation_pressure         * 0.10
    )

    return {
        "pressure":          round(pressure,          3),
        "attraction":        round(attraction,        3),
        "inhibition":        round(inhibition,        3),
        "control_strain":    round(control_strain,    3),
        "exposure":          round(exposure,          3),
        "approach_drive":    round(approach_drive,    3),
        "conflict_load":     round(conflict_load,     3),
        # interpretable sub-drivers (useful for audit / future UI)
        "price_sensitivity": round(price_sensitivity, 3),
        "world_shame":       round(world_shame,       3),
        "bond_conflict":     round(bond_conflict,     3),
        "violation_pressure":round(violation_pressure,3),
    }


# ---------------------------------------------------------------------------
# Pressure band  (per-action HIGH threshold, computed once)
# ---------------------------------------------------------------------------

def resolve_pressure_band(pressure: float, selected_action: str) -> str:
    high_thresh = HIGH_THRESHOLDS.get(selected_action, 0.55)
    if pressure >= high_thresh:
        return "HIGH"
    if pressure >= MID_THRESHOLD:
        return "MID"
    return "LOW"


# ---------------------------------------------------------------------------
# Object anchor helper
# ---------------------------------------------------------------------------

def choose_object_anchor(scene_affordances: Mapping[str, Any],
                          default_behavior: str) -> str:
    objects = scene_affordances.get("objects_available", []) or []
    if "drink_glass" in objects:
        return default_behavior
    if "phone" in objects:
        return "object_play"
    if "bar_counter" in objects:
        return "barrier_object"
    return "none"


# ---------------------------------------------------------------------------
# Action-specific token compilers
# All receive (drivers, scene_affordances, pressure_band) — band computed once.
# ---------------------------------------------------------------------------

ActionCompiler = Callable[
    [Mapping[str, float], Mapping[str, Any], str],
    Dict[str, str]
]


def _compile_flirt_no_commit(
    drivers: Mapping[str, float],
    scene_affordances: Mapping[str, Any],
    pressure_band: str,
) -> Dict[str, str]:
    attraction = drivers["attraction"]
    inhibition = drivers["inhibition"]

    if pressure_band == "HIGH" or (attraction > 0.60 and inhibition > 0.75):
        return {
            "gaze_contact":       "brief",
            "smile_state":        "micro",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "tight_grip"),
            "response_latency":   "hesitation",
            "voice_texture":      "warm_restrained",
            "distance_behavior":  "holds_distance",
            "movement_quality":   "approach_then_check",
        }

    if pressure_band == "MID" or (attraction > 0.60 and inhibition > 0.50):
        return {
            "gaze_contact":       "intermittent",
            "smile_state":        "emerging_then_checked",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "anchoring_on_glass"),
            "response_latency":   "short_pause",
            "voice_texture":      "warm_restrained",
            "distance_behavior":  "holds_distance",
            "movement_quality":   "approach_then_check",
        }

    return {
        "gaze_contact":       "held",
        "smile_state":        "warm_full",
        "torso_openness":     "open",
        "object_interaction": "loose_hold",
        "response_latency":   "immediate",
        "voice_texture":      "open_charismatic",
        "distance_behavior":  "partial_approach",
        "movement_quality":   "fluid",
    }


def _compile_change_subject(
    drivers: Mapping[str, float],
    scene_affordances: Mapping[str, Any],
    pressure_band: str,
) -> Dict[str, str]:
    if pressure_band == "HIGH":
        return {
            "gaze_contact":       "avoidant",
            "smile_state":        "tight_mask",
            "torso_openness":     "closed",
            "object_interaction": choose_object_anchor(scene_affordances, "barrier_object"),
            "response_latency":   "immediate",
            "voice_texture":      "measured",
            "distance_behavior":  "retreat",          # was holds_distance in v0.2-A
            "movement_quality":   "still",
        }

    if pressure_band == "MID":
        return {
            "gaze_contact":       "brief",
            "smile_state":        "micro",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "loose_hold"),
            "response_latency":   "short_pause",
            "voice_texture":      "measured",
            "distance_behavior":  "holds_distance",
            "movement_quality":   "still",
        }

    return {
        "gaze_contact":       "intermittent",
        "smile_state":        "none",
        "torso_openness":     "partial",
        "object_interaction": "none",
        "response_latency":   "immediate",
        "voice_texture":      "flat",
        "distance_behavior":  "holds_distance",
        "movement_quality":   "still",
    }


def _compile_accept_invitation(
    drivers: Mapping[str, float],
    scene_affordances: Mapping[str, Any],
    pressure_band: str,
) -> Dict[str, str]:
    inhibition = drivers["inhibition"]

    if pressure_band == "HIGH" or inhibition > 0.70:
        return {
            "gaze_contact":       "intermittent",
            "smile_state":        "emerging_then_checked",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "anchoring_on_glass"),
            "response_latency":   "hesitation",
            "voice_texture":      "warm_restrained",
            "distance_behavior":  "partial_approach",
            "movement_quality":   "approach_then_check",
        }

    if pressure_band == "MID":                        # new in v0.3
        return {
            "gaze_contact":       "held",
            "smile_state":        "micro",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "loose_hold"),
            "response_latency":   "short_pause",
            "voice_texture":      "warm_restrained",
            "distance_behavior":  "partial_approach",
            "movement_quality":   "approach_then_check",
        }

    return {
        "gaze_contact":       "held",
        "smile_state":        "warm_full",
        "torso_openness":     "open",
        "object_interaction": "loose_hold",
        "response_latency":   "immediate",
        "voice_texture":      "open_charismatic",
        "distance_behavior":  "approach",
        "movement_quality":   "fluid",
    }


def _compile_confront_married(
    drivers: Mapping[str, float],
    scene_affordances: Mapping[str, Any],
    pressure_band: str,
) -> Dict[str, str]:
    if pressure_band == "HIGH":
        return {
            "gaze_contact":       "locked",
            "smile_state":        "none",
            "torso_openness":     "open",
            "object_interaction": choose_object_anchor(scene_affordances, "tight_grip"),
            "response_latency":   "long_hold",
            "voice_texture":      "raw_breaking",
            "distance_behavior":  "holds_distance",
            "movement_quality":   "frozen",
        }

    if pressure_band == "MID":
        return {
            "gaze_contact":       "held",
            "smile_state":        "none",
            "torso_openness":     "partial",
            "object_interaction": choose_object_anchor(scene_affordances, "anchoring_on_glass"),
            "response_latency":   "hesitation",
            "voice_texture":      "measured",
            "distance_behavior":  "holds_distance",
            "movement_quality":   "still",
        }

    return {
        "gaze_contact":       "brief",
        "smile_state":        "none",
        "torso_openness":     "partial",
        "object_interaction": "none",
        "response_latency":   "short_pause",
        "voice_texture":      "flat",
        "distance_behavior":  "holds_distance",
        "movement_quality":   "still",
    }


def _compile_withdraw(
    drivers: Mapping[str, float],
    scene_affordances: Mapping[str, Any],
    pressure_band: str,
) -> Dict[str, str]:
    return {
        "gaze_contact":       "avoidant",
        "smile_state":        "none",
        "torso_openness":     "closed",
        "object_interaction": choose_object_anchor(scene_affordances, "barrier_object"),
        "response_latency":   "immediate",
        "voice_texture":      "flat",
        "distance_behavior":  "retreat",
        "movement_quality":   "still",
    }


ACTION_COMPILERS: Dict[str, ActionCompiler] = {
    "flirt_no_commit":   _compile_flirt_no_commit,
    "change_subject":    _compile_change_subject,
    "accept_invitation": _compile_accept_invitation,
    "confront_married":  _compile_confront_married,
    "withdraw":          _compile_withdraw,
}

# Intersection: only actions with BOTH metadata AND a compiler.
# Prevents a future ACTION_META entry without a compiler from being flagged as implemented.
IMPLEMENTED_ACTIONS = set(ACTION_META.keys()) & set(ACTION_COMPILERS.keys())


# ---------------------------------------------------------------------------
# Token → render text
# ---------------------------------------------------------------------------

TOKEN_TEXT_MAP: Dict[str, Dict[str, str]] = {
    "gaze_contact": {
        "avoidant":    "His gaze stays away from her, settling on the bar or his drink.",
        "brief":       "He meets her eyes only briefly, then looks away.",
        "intermittent":"He holds her gaze for a moment, then lets it drop before the exchange becomes too intimate.",
        "held":        "He holds direct eye contact with her in a natural, confident way.",
        "locked":      "His gaze stays fixed and intense.",
    },
    "smile_state": {
        "none":                  "No smile appears.",
        "micro":                 "A very small smile appears and disappears quickly.",
        "emerging_then_checked": "A small smile starts, then checks itself before fully settling.",
        "warm_full":             "The smile lands easily and stays natural.",
        "tight_mask":            "The smile looks controlled, more mask than ease.",
    },
    "torso_openness": {
        "closed":  "His body stays closed and guarded.",
        "partial": "His body turns toward her, but remains only partly open, keeping a subtle exit.",
        "open":    "His body is open and fully oriented toward her.",
    },
    "object_interaction": {
        "none":               "",   # omitted by OMIT_NONE_CHANNELS
        "loose_hold":         "His hand rests loosely near the drink.",
        "object_play":        "He lightly plays with the object without tension.",
        "anchoring_on_glass": "His hand stays on the glass a moment longer than necessary, using it as a small anchor.",
        "tight_grip":         "His hand grips the glass with visible tension.",
        "barrier_object":     "The object remains between them like a small barrier.",
    },
    "response_latency": {
        "immediate":  "He answers immediately.",
        "short_pause":"There is a short pause before he answers.",
        "hesitation": "He hesitates before answering.",
        "long_hold":  "A long silence hangs before the words arrive.",
    },
    "voice_texture": {
        "flat":            "His delivery is flat and factual.",
        "measured":        "His delivery is controlled and measured.",
        "warm_restrained": "His delivery is warm, but restrained.",
        "open_charismatic":"His delivery is easy, playful, and confident.",
        "raw_breaking":    "His delivery carries strain, as if the words cost him.",
    },
    "distance_behavior": {
        "retreat":         "He leans or turns slightly away.",
        "holds_distance":  "He maintains the distance between them.",
        "partial_approach":"He shifts fractionally closer without closing the gap.",
        "approach":        "He moves comfortably into the exchange.",
    },
    "movement_quality": {
        "still":              "His body remains still, holding the moment.",
        "fluid":              "The movement feels fluid and socially comfortable.",
        "approach_then_check":"He almost leans in, then checks himself.",
        "restless":           "Small restless movements leak through the body.",
        "frozen":             "His body seems to freeze before the action completes.",
    },
}

# Render order: spatial (torso/distance) → kinetic (movement) → object → face → gaze → timing → voice
RENDER_ORDER: List[str] = [
    "torso_openness",
    "distance_behavior",
    "movement_quality",
    "object_interaction",
    "smile_state",
    "gaze_contact",
    "response_latency",
    "voice_texture",
]

# Channels where value="none" is silently omitted rather than rendered.
# object_interaction=none → no line. smile_state=none → "No smile appears."
OMIT_NONE_CHANNELS = {"object_interaction"}


def tokens_to_render_lines(tokens: Mapping[str, str]) -> List[str]:
    lines: List[str] = []
    for channel in RENDER_ORDER:
        value = tokens.get(channel)
        if not value:
            continue
        if value == "none" and channel in OMIT_NONE_CHANNELS:
            continue
        text = TOKEN_TEXT_MAP.get(channel, {}).get(value, "")
        if text:
            lines.append(text)
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_behavioral_spec(
    payload: Mapping[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point.

    Args:
        payload:  SSV, NSV, World, selected_action, score_gap, scene_affordances.
        strict:   True  → raise ValueError on unknown action (dev/test mode).
                  False → production fallback to 'withdraw' with flag in debug.

    Returns:
        drivers, pressure_band, behavioral_tokens, render_lines, debug.
    """
    selected_action   = str(payload.get("selected_action", "unknown_action"))
    score_gap         = float(payload.get("score_gap", 0.0))
    ssv               = get_nested(payload, "ssv")
    nsv               = get_nested(payload, "nsv")
    world             = get_nested(payload, "world")
    scene_affordances = get_nested(payload, "scene_affordances")

    is_implemented = selected_action in IMPLEMENTED_ACTIONS

    if not is_implemented:
        if strict:
            raise ValueError(
                f"No compiler for action '{selected_action}'. "
                f"Implemented: {sorted(IMPLEMENTED_ACTIONS)}"
            )
        action_meta = ACTION_META["withdraw"]
        compiler_fn = _compile_withdraw
    else:
        action_meta = ACTION_META[selected_action]
        compiler_fn = ACTION_COMPILERS[selected_action]

    drivers       = compute_behavioral_drivers(ssv, nsv, world, action_meta, score_gap)
    pressure_band = resolve_pressure_band(drivers["pressure"], selected_action)
    tokens        = compiler_fn(drivers, scene_affordances, pressure_band)
    render_lines  = tokens_to_render_lines(tokens)

    return {
        "drivers":           drivers,
        "pressure_band":     pressure_band,
        "behavioral_tokens": tokens,
        "render_lines":      render_lines,
        "debug": {
            "selected_action":    selected_action,
            "score_gap":          round(score_gap, 3),
            "implemented_action": is_implemented,
            "fallback_action":    None if is_implemented else "withdraw",
            "compiler_version":   COMPILER_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------

def render_prompt_block(render_lines: List[str]) -> str:
    """LTX format: PERFORMANCE header + one directive per line."""
    return "PERFORMANCE:\n" + "\n".join(render_lines)


def build_performance_paragraph(compiled: Mapping[str, Any]) -> str:
    """Runway / inline format: single flowing paragraph."""
    return " ".join(compiled.get("render_lines", []))
