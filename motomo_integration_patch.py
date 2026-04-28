"""
MoToMo Integration Patch
Connects motomo_engine.py → motomo_behavioral_compiler.py

What this file does:
  1. SSV adapter:  engine SSV dataclass → compiler SSV dict
  2. NSV adapter:  engine NSV dataclass → compiler NSV dict (with P_imm / P_lat)
  3. World + scene defaults for the POC bar scene
  4. Patched build_ltx_prompt() that uses compile_behavioral_spec()
     instead of the old hardcoded posture/voice/pace/gaze specs

What this file does NOT change:
  - compute_rankings()
  - apply_nsv_update()
  - score_action()
  - compute_utility()
  - All the drama engine math — untouched

How to integrate:
  In motomo_engine.py:
    1. from motomo_integration_patch import build_ltx_prompt
    OR inline the functions below directly into motomo_engine.py
       and remove the old build_ltx_prompt + compute_performance_spec.
"""

import importlib.util, pathlib, sys
from typing import Dict, List, Optional

# ── Load compiler ─────────────────────────────────────────────────────────────
# Assumes motomo_behavioral_compiler.py is in the same directory.
_compiler_path = pathlib.Path(__file__).parent / "motomo_behavioral_compiler.py"
_spec = importlib.util.spec_from_file_location("behavioral_compiler", _compiler_path)
_compiler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compiler)

compile_behavioral_spec = _compiler.compile_behavioral_spec
render_prompt_block     = _compiler.render_prompt_block


# ── POC world constants (realistic marital drama) ────────────────────────────
POC_WORLD = {
    "M": 0.72,   # moral strictness
    "S": 0.34,   # social surveillance (bar = low visibility)
    "B": 0.84,   # bond sacredness (marriage)
    "H": 0.38,
    "V": 0.28,
    "F": 0.56,
    "T": 0.42,
}

POC_SCENE_BAR = {
    "location": "bar",
    "objects_available": ["drink_glass", "bar_counter", "empty_stool"],
    "interaction_partner": "Maya",
    "body_position": "seated",
}


# ── SSV adapter ──────────────────────────────────────────────────────────────

def ssv_to_compiler_dict(ssv) -> Dict[str, float]:
    """
    Maps engine SSV dataclass → compiler SSV dict.

    Engine fields         Compiler fields   Notes
    ─────────────────     ───────────────   ─────────────────────────────────
    RT  (risk tolerance)  R                 direct
    loyalty               W                 loyalty-to-code / world-rules
    DT  (dominance)       C                 control / agency need
    1 - PS (price sens.)  P                 PS is sensitivity; P is tolerance
    validation            D                 desire / need pressure (proxy)
    loyalty               A                 loyalty-to-people / relational bond

    Note: the engine uses a single 'loyalty' field.
    The compiler splits it into W (world-code) and A (relational).
    For POC we map both from the same source value.
    When the engine is updated to the new 6-variable SSV, remove this adapter.
    """
    price_tolerance = max(0.0, min(1.0, 1.0 - ssv.PS))

    return {
        "R": ssv.RT,                    # risk orientation
        "W": ssv.loyalty,               # world-code sensitivity
        "C": ssv.DT,                    # control / agency need
        "P": price_tolerance,           # price tolerance (inverted from PS)
        "D": ssv.validation,            # desire / need pressure
        "A": ssv.loyalty,               # relational binding (same source for POC)
    }


# ── NSV adapter ──────────────────────────────────────────────────────────────

def nsv_to_compiler_dict(nsv, score_gap: float = 0.0) -> Dict[str, float]:
    """
    Maps engine NSV dataclass → compiler NSV dict.

    Engine fields    Compiler fields   Notes
    ─────────────    ───────────────   ──────────────────────────────────────
    E                E                 direct (engine: [-1,1], compiler: [0,1])
    C                C                 direct (same range caveat)
    R                R                 direct
    P                P_imm             immediate price felt right now
    Du               P_lat             distance from unconscious goal ≈ latent price
    Dc               D                 distance from conscious goal
    Yu               Y                 unconscious goal activation
    —                W                 world pressure: derived from R + P_imm
    """
    # Normalise E and C from [-1,1] to [0,1] for the compiler
    E_norm = max(0.0, min(1.0, (nsv.E + 1.0) / 2.0))
    C_norm = max(0.0, min(1.0, (nsv.C + 1.0) / 2.0))

    # World pressure proxy: high risk + high immediate price = world closing in
    W_proxy = max(0.0, min(1.0, nsv.R * 0.5 + nsv.P * 0.5))

    return {
        "E":     E_norm,
        "C":     C_norm,
        "R":     nsv.R,
        "P_imm": nsv.P,                  # immediate felt price
        "P_lat": max(0.0, min(1.0, nsv.Du)),   # latent price from unconscious distance
        "W":     W_proxy,
        "D":     max(0.0, min(1.0, nsv.Dc)),
        "Y":     nsv.Yu,
    }


# ── Camera specs (unchanged from engine) ─────────────────────────────────────

CAMERA_SPECS = {
    "accept_invitation": "Two-shot, eye-level, medium depth, slow push-in toward Maya",
    "flirt_no_commit":   "Two-shot, eye-level, medium depth, conversational rhythm",
    "change_subject":    "Medium shot, slight rack focus, character turns body",
    "confront_married":  "Tight close-up, static frame, very long take, character faces camera",
}


# ── Patched build_ltx_prompt ─────────────────────────────────────────────────

def build_ltx_prompt(
    character_name: str,
    top_action: Dict,
    ssv,                        # engine SSV dataclass
    nsv,                        # engine NSV dataclass
    signal,                     # engine ExternalSignal
    beat_context: str,
    score_gap: float = 0.0,
    selection_type: str = "recommended",
    recommended_action: Optional[Dict] = None,
    scene_affordances: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Patched version of build_ltx_prompt.
    PERFORMANCE block is now generated by compile_behavioral_spec()
    instead of the old hardcoded posture/voice/pace/gaze templates.
    Everything else (audit line, camera, NSV context, duration) unchanged.
    """
    action = top_action["action"]

    # ── Convert SSV / NSV to compiler format ─────────────────────────────
    compiler_ssv = ssv_to_compiler_dict(ssv)
    compiler_nsv = nsv_to_compiler_dict(nsv, score_gap)

    # ── Build compiler payload ────────────────────────────────────────────
    compiler_payload = {
        "selected_action":   action,
        "score_gap":         score_gap,
        "ssv":               compiler_ssv,
        "nsv":               compiler_nsv,
        "world":             POC_WORLD,
        "scene_affordances": scene_affordances or POC_SCENE_BAR,
    }

    compiled = compile_behavioral_spec(compiler_payload, strict=False)
    performance_block = render_prompt_block(compiled["render_lines"])
    pressure_band     = compiled["pressure_band"]

    # ── NSV context line (unchanged logic) ───────────────────────────────
    if signal.wife_message_active:
        nsv_context = (
            f"[wife_message: Active — relational_commitment_salience=0.91] "
            f"Character is under relational pressure. "
            f"band={pressure_band}."
        )
    else:
        nsv_context = (
            f"[wife_message: Inactive — relational_commitment_salience=0.15] "
            f"No relational constraint active. "
            f"band={pressure_band}."
        )

    # ── Audit line (unchanged logic) ─────────────────────────────────────
    if selection_type == "override" and recommended_action:
        resistance_note = _resistance_note(score_gap)
        audit_line = (
            f"MoToMo selected_action={action} by creator override. "
            f"Engine recommended {recommended_action['action']}. "
            f"Score gap={score_gap:.3f}; {resistance_note} [{pressure_band} pressure]."
        )
    else:
        audit_line = (
            f"MoToMo score={top_action['score']:.3f} "
            f"[loyalty={ssv.loyalty:.2f}, shame={ssv.shame:.2f}, RT={ssv.RT:.2f}] "
            f"computed {action} as most internally consistent given NSV."
        )

    # ── Assemble render-clean prompt ──────────────────────────────────────
    # PERFORMANCE block comes from compiler — no engine metadata inside it.
    compiled_prompt = (
        f"[MoToMo-v4.3 | Behavioral Compiler v0.3.1]\n"
        f"[Character: {character_name}]\n"
        f"[Beat: Uri at the Bar — Beat 7]\n"
        f"[Location: Bar — Late Evening]\n"
        f"[{beat_context}]\n\n"
        f"ACTION: {top_action['label']}\n\n"
        f"{performance_block}\n\n"
        f"CAMERA: {CAMERA_SPECS.get(action, 'Two-shot, eye-level')}\n\n"
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
        "compiled_prompt":   compiled_prompt,
        "negative_prompt":   negative_prompt,
        "camera":            CAMERA_SPECS.get(action, "Two-shot, eye-level"),
        "pressure_band":     pressure_band,
        "render_lines":      compiled["render_lines"],      # ← new: frontend can use directly
        "behavioral_tokens": compiled["behavioral_tokens"], # ← new: for UI/audit
        "drivers":           compiled["drivers"],           # ← new: for audit only
        "vocal_tone":        compiled["behavioral_tokens"].get("voice_texture", ""),
        "duration":          9 if action != "confront_married" else 12,
        "aspect_ratio":      "2.39:1",
        "resolution":        "1920x804",
    }


def _resistance_note(score_gap: float) -> str:
    if score_gap < 0.08:
        return "performance remains fluid — minimal internal resistance"
    if score_gap < 0.25:
        return "subtle internal resistance visible in timing and gaze"
    if score_gap < 0.45:
        return "visible resistance — body holds cost of the override"
    return "rupture — body contradicts the action"
