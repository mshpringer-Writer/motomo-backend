"""
MoToMo Backend — Final POC
===========================
Endpoints:
  GET  /health
  POST /compute-ab            — action rankings A/B (no cinematic layer)
  POST /compute-cinematic-ab  — full pipeline: rankings + v2 cinematic delta
  POST /generate              — fal.ai LTX video generation (A/B)
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Local engines ──────────────────────────────────────────────────────────
from motomo_engine import (
    SSV, NSV, ExternalSignal,
    compute_rankings as _compute_rankings,
    apply_nsv_update,
)
from motomo_integration_patch import build_ltx_prompt
from motomo_cinematic_v2 import (
    CinematicEngine, PromptCompiler,
    FILTERS as DSV_FILTERS,
    NSV as CinNSV,
)

app = FastAPI(title="MoToMo POC Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine   = CinematicEngine()
compiler = PromptCompiler()

BASE_DURATIONS = {
    "WITNESS": 16.0,
    "PRISM":   12.0,
    "DISSOLVE":13.0,
    "CURRENT": 6.0,
    "SIGNAL":  14.0,
}

FILTER_TAGLINES = {
    "WITNESS":  "The camera watches. It does not intervene.",
    "PRISM":    "Geometric. Cold. The world does not bend for feeling.",
    "DISSOLVE": "Time bends. The camera is inside longing.",
    "CURRENT":  "The world presses forward. The camera is caught in it.",
    "SIGNAL":   "Something has changed that cannot be undone.",
}


# ── Pydantic Models ────────────────────────────────────────────────────────

class SSVInput(BaseModel):
    loyalty_sensitivity:  float = 0.82
    validation_need:      float = 0.78
    risk_tolerance:       float = 0.22
    shame_sensitivity:    float = 0.88
    conflict_avoidance:   float = 0.82

class CinematicRequest(BaseModel):
    character_name:        str   = "Uri — The Wounded Loyalist"
    beat:                  str   = "Beat 7 — Bar, Late Evening"
    scene_description:     str   = "Maya invites him home"
    ssv:                   SSVInput
    filter_name:           str   = "DISSOLVE"
    flirtation_intensity:  float = 0.75

# For fal.ai video generation
class LTXPackage(BaseModel):
    compiled_prompt:  str
    negative_prompt:  str
    duration:         float = 9.0
    aspect_ratio:     str   = "2.39:1"
    resolution:       str   = "1920x804"
    character_id:     str
    action:           str

class NSVScenario(BaseModel):
    label:       str
    nsv_signal:  str
    ltx_package: LTXPackage

class GenerateRequest(BaseModel):
    character_name:    str
    beat:              str
    scene_description: str
    scenario_a:        NSVScenario
    scenario_b:        NSVScenario


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_ssv(s: SSVInput) -> SSV:
    return SSV(
        RT=s.risk_tolerance,
        PS=s.shame_sensitivity,
        DT=max(0.0, 1.0 - s.conflict_avoidance),
        loyalty=s.loyalty_sensitivity,
        shame=s.shame_sensitivity,
        validation=s.validation_need,
        conflict_avoidance=s.conflict_avoidance,
    )

def _run_scenario(ssv: SSV, wife_message: bool, flirt: float):
    base_nsv = NSV(E=0.3, C=0.2, R=0.3, P=0.2, Dc=0.5, Du=0.6)
    signal   = ExternalSignal(wife_message_active=wife_message,
                               flirtation_intensity=flirt)
    rankings = _compute_rankings(ssv, base_nsv, signal)
    nsv_out  = apply_nsv_update(base_nsv, ssv, signal)
    top      = rankings[0]
    ltx      = build_ltx_prompt("Uri", top, ssv, nsv_out, signal, "Maya invites him home")
    return rankings, nsv_out, top, ltx


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "engine": "MoToMo-v4.2"}


@app.post("/compute-ab")
def compute_ab(req: CinematicRequest):
    """
    Action rankings only — no cinematic layer.
    Quick endpoint for testing the behavioral engine.
    """
    ssv = _build_ssv(req.ssv)
    ra, nsv_a, top_a, ltx_a = _run_scenario(ssv, False, req.flirtation_intensity)
    rb, nsv_b, top_b, ltx_b = _run_scenario(ssv, True,  req.flirtation_intensity)

    sa = {r["action"]: r["score"] for r in ra}
    sb = {r["action"]: r["score"] for r in rb}
    deltas = {a: round(sb.get(a,0) - sa.get(a,0), 3) for a in sa}

    return {
        "character": req.character_name,
        "beat":      req.beat,
        "scenario_a": {
            "label": "No wife message",
            "nsv":   {"E": round(nsv_a.E,3), "C": round(nsv_a.C,3),
                      "R": round(nsv_a.R,3), "P": round(nsv_a.P,3)},
            "rankings":   ra,
            "top_action": top_a["action"],
            "top_score":  top_a["score"],
            "ltx_prompt": ltx_a["compiled_prompt"],
        },
        "scenario_b": {
            "label": "Wife message arrives",
            "nsv":   {"E": round(nsv_b.E,3), "C": round(nsv_b.C,3),
                      "R": round(nsv_b.R,3), "P": round(nsv_b.P,3)},
            "rankings":   rb,
            "top_action": top_b["action"],
            "top_score":  top_b["score"],
            "ltx_prompt": ltx_b["compiled_prompt"],
        },
        "score_deltas":    deltas,
        "narrative_shift": f"{top_a['action']} → {top_b['action']}",
    }


@app.post("/compute-cinematic-ab")
def compute_cinematic_ab(req: CinematicRequest):
    """
    Full pipeline:
      1. Action rankings A + B
      2. v2 cinematic delta (engine + compiler)
      3. Render tokens per filter
    """
    fname = req.filter_name.upper()
    if fname not in DSV_FILTERS:
        raise HTTPException(400, f"Unknown filter '{fname}'. Options: {list(DSV_FILTERS)}")

    dsv  = DSV_FILTERS[fname]
    ssv  = _build_ssv(req.ssv)
    base = BASE_DURATIONS[fname]

    ra, nsv_a, top_a, ltx_a = _run_scenario(ssv, False, req.flirtation_intensity)
    rb, nsv_b, top_b, ltx_b = _run_scenario(ssv, True,  req.flirtation_intensity)

    # v2 cinematic delta
    cin_a = CinNSV(E=nsv_a.E, C=nsv_a.C, R=nsv_a.R, P=nsv_a.P)
    cin_b = CinNSV(E=nsv_b.E, C=nsv_b.C, R=nsv_b.R, P=nsv_b.P)
    decision = engine.build_decision(cin_a, cin_b, dsv, base)
    tokens   = compiler.compile_render_tokens(decision)
    human    = compiler.compile_human(decision)
    render   = compiler.compile_render_prompt(decision)

    # Enrich LTX prompts with cinematic layer
    ltx_a_enriched = ltx_a["compiled_prompt"] + (
        f"\n\n[{fname} — SCENARIO A]\n"
        f"{', '.join(dsv.anchor_tokens)}\n"
        f"duration: {base:.0f}s"
    )
    ltx_b_enriched = ltx_b["compiled_prompt"] + (
        f"\n\n[{fname} — SCENARIO B — CINEMATIC DELTA]\n"
        f"{render}"
    )

    sa = {r["action"]: r["score"] for r in ra}
    sb = {r["action"]: r["score"] for r in rb}
    deltas = {a: round(sb.get(a,0) - sa.get(a,0), 3) for a in sa}

    return {
        "character":      req.character_name,
        "beat":           req.beat,
        "filter":         fname,
        "filter_tagline": FILTER_TAGLINES[fname],

        "scenario_a": {
            "label":      "No wife message",
            "nsv":        {"E":round(nsv_a.E,3),"C":round(nsv_a.C,3),
                           "R":round(nsv_a.R,3),"P":round(nsv_a.P,3)},
            "rankings":   ra,
            "top_action": top_a["action"],
            "top_score":  top_a["score"],
            "ltx_prompt": ltx_a_enriched,
        },
        "scenario_b": {
            "label":      "Wife message arrives",
            "nsv":        {"E":round(nsv_b.E,3),"C":round(nsv_b.C,3),
                           "R":round(nsv_b.R,3),"P":round(nsv_b.P,3)},
            "rankings":   rb,
            "top_action": top_b["action"],
            "top_score":  top_b["score"],
            "ltx_prompt": ltx_b_enriched,
        },

        "score_deltas":    deltas,
        "narrative_shift": f"{top_a['action']} → {top_b['action']}",

        "cinematic_delta": {
            "human_explanation": human,
            "render_tokens":     tokens,
            "render_prompt":     render,
            "framing":   {"angle": decision.framing.angle,
                          "composition": decision.framing.composition,
                          "dominance": decision.framing.dominance},
            "lens":      {"mm": decision.lens.focal_length_mm,
                          "aperture": decision.lens.aperture_hint},
            "movement":  {"mode": decision.movement.mode,
                          "cuts_added": decision.movement.cut_delta},
            "duration":  {"scenario_a_s": base,
                          "scenario_b_s": decision.duration.final_seconds,
                          "hold_ext": decision.duration.hold_extension},
            "lighting":  {"mode": decision.lighting.mode,
                          "kelvin": decision.lighting.kelvin_hint},
            "depth":     {"mode": decision.depth.mode},
            "subjectivity": {"position": decision.subjectivity.final_position},
            "special_event": {
                "enabled": decision.special_event.enabled,
                "delay_s": decision.special_event.delay_seconds,
                "mm":      decision.special_event.magnitude_mm,
            } if decision.special_event.enabled else None,
        },
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """fal.ai LTX video generation for A/B scenarios."""
    try:
        import fal_client
    except ImportError:
        raise HTTPException(500, "fal-client not installed")

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise HTTPException(500, "FAL_KEY not set")
    os.environ["FAL_KEY"] = fal_key

    async def gen_one(scenario: NSVScenario):
        w, h = map(int, scenario.ltx_package.resolution.split("x"))
        def _call():
            return fal_client.run(
                "fal-ai/ltx-video",
                arguments={
                    "prompt":           scenario.ltx_package.compiled_prompt,
                    "negative_prompt":  scenario.ltx_package.negative_prompt,
                    "num_frames":       int(scenario.ltx_package.duration * 25),
                    "width": w, "height": h,
                    "guidance_scale":   3.5,
                    "num_inference_steps": 40,
                    "seed": 42,
                }
            )
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _call)
        return {"label": scenario.label, "video_url": result["video"]["url"],
                "action": scenario.ltx_package.action}

    try:
        a, b = await asyncio.gather(gen_one(req.scenario_a), gen_one(req.scenario_b))
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"character": req.character_name, "beat": req.beat,
            "scenario_a": a, "scenario_b": b,
            "narrative_shift": f"{a['action']} → {b['action']}"}
# ─── Override Endpoint ────────────────────────────────────────────────────────

class OverrideRequest(BaseModel):
    character_name:      str   = "Uri — The Wounded Loyalist"
    beat:                str   = "Beat 7 — Bar, Late Evening"
    scene_description:   str   = "Maya invites him home"
    ssv:                 SSVInput
    filter_name:         str   = "DISSOLVE"
    wife_message_active: bool  = False
    chosen_action:       str
    top_action:          str
    top_score:           float
    chosen_score:        float


@app.post("/compute-override")
def compute_override(req: OverrideRequest):
    score_gap = round(req.top_score - req.chosen_score, 3)

    if score_gap > 0.40:
        level          = "red"
        forced_filter  = "SIGNAL"
        warning        = (
            f"This action exceeds the character's computed threshold by {score_gap:.3f}. "
            f"Engine recommended '{req.top_action}' ({req.top_score:.3f}). "
            f"Choosing '{req.chosen_action}' ({req.chosen_score:.3f}) is a character rupture."
        )
        cinematic_note = (
            "CHARACTER RUPTURE — action violates internal logic. "
            "Cinematic treatment: SIGNAL filter forced. "
            "High contrast revelation light. Held threshold frame. "
            "Delayed reframe on the moment of decision."
        )
    elif score_gap > 0.20:
        level          = "yellow"
        forced_filter  = req.filter_name.upper()
        warning        = (
            f"Borderline action. Gap: {score_gap:.3f}. "
            f"Engine preferred '{req.top_action}' ({req.top_score:.3f}). "
            f"'{req.chosen_action}' ({req.chosen_score:.3f}) is plausible with intention."
        )
        cinematic_note = (
            "BORDERLINE ACTION — use with dramatic intention. "
            "Slightly harder contrast, longer hold post-action."
        )
    else:
        level          = "green"
        forced_filter  = req.filter_name.upper()
        warning        = (
            f"Action within acceptable range. Gap: {score_gap:.3f}. "
            f"Override is minimal — character logic holds."
        )
        cinematic_note = "Action is consistent with character. No cinematic adjustment needed."

    fname = forced_filter if forced_filter in DSV_FILTERS else "SIGNAL"
    dsv   = DSV_FILTERS[fname]
    ssv   = _build_ssv(req.ssv)
    base  = BASE_DURATIONS[fname]

    _, nsv_a, _, _ = _run_scenario(ssv, False, 0.75)
    _, nsv_b, _, _ = _run_scenario(ssv, req.wife_message_active, 0.75)

    cin_a    = CinNSV(E=nsv_a.E, C=nsv_a.C, R=nsv_a.R, P=nsv_a.P)
    cin_b    = CinNSV(E=nsv_b.E, C=nsv_b.C, R=nsv_b.R, P=nsv_b.P)
    decision = engine.build_decision(cin_a, cin_b, dsv, base)
    tokens   = compiler.compile_render_tokens(decision)
    render   = compiler.compile_render_prompt(decision)

    # ── Build prompt from CHOSEN action — not recommended ──
    ACTION_LABELS = {
        "accept_invitation": "Accept invitation",
        "flirt_no_commit":   "Flirt without commitment",
        "change_subject":    "Change subject",
        "confront_married":  '"I\'m married" — direct confrontation',
    }

    chosen_action_obj = {
        "action": req.chosen_action,
        "label":  ACTION_LABELS.get(req.chosen_action, req.chosen_action),
        "score":  req.chosen_score,
    }
  # ── Behavioral Compiler on selected_action (not top-ranked) ──
        from motomo_integration_patch import (
            ssv_to_compiler_dict, nsv_to_compiler_dict,
            POC_WORLD, POC_SCENE_BAR, compile_behavioral_spec,
        )
        _compiler_payload = {
            "selected_action":   req.chosen_action,
            "score_gap":         score_gap,
            "ssv":               ssv_to_compiler_dict(ssv),
            "nsv":               nsv_to_compiler_dict(nsv_b),
            "world":             POC_WORLD,
            "scene_affordances": POC_SCENE_BAR,
        }
        _compiled = compile_behavioral_spec(_compiler_payload, strict=False)

    signal = ExternalSignal(
        wife_message_active=req.wife_message_active,
        flirtation_intensity=0.75,
    )

    ltx_chosen = build_ltx_prompt(
    req.character_name,
    chosen_action_obj,
    ssv,
    nsv_b,
    signal,
    req.scene_description,
    score_gap=score_gap,
    selection_type="override",
    recommended_action={"action": req.top_action, "score": req.top_score},
)

    override_header = (
        f"[MOTOMO OVERRIDE — {level.upper()}]\n"
        f"[Character: {req.character_name}]\n"
        f"[Beat: {req.beat}]\n"
        f"[SELECTED ACTION: {req.chosen_action} — score: {req.chosen_score:.3f}]\n"
        f"[ENGINE RECOMMENDED: {req.top_action} — score: {req.top_score:.3f}]\n"
        f"[SCORE GAP: {score_gap:.3f}]\n\n"
        f"{cinematic_note}\n\n"
        f"{ltx_chosen['compiled_prompt']}\n\n"
        f"[{fname} FILTER]\n"
        f"{render}"
    )

    return {
        "override_level":  level,
        "score_gap":       score_gap,
        "chosen_action":   req.chosen_action,
        "chosen_score":    req.chosen_score,
        "top_action":      req.top_action,
        "top_score":       req.top_score,
        "warning":         warning,
        "forced_filter":   fname,
        "filter_tagline":  FILTER_TAGLINES[fname],
        "cinematic_note":  cinematic_note,
        "render_tokens":   tokens,
        "render_prompt":   render,
        "ltx_prompt":      override_header,
      "behavioral_compiler": {
            "render_lines":      _compiled["render_lines"],
            "behavioral_tokens": _compiled["behavioral_tokens"],
            "pressure_band":     _compiled["pressure_band"],
            "drivers":           _compiled["drivers"],
            "debug":             _compiled["debug"],
        },
        "render_lines":      _compiled["render_lines"],
        "behavioral_tokens": _compiled["behavioral_tokens"],
        "pressure_band":     _compiled["pressure_band"],
        "drivers":           _compiled["drivers"],
        "cinematic_delta": {
            "framing":   {"angle": decision.framing.angle,
                          "composition": decision.framing.composition},
            "lens":      {"mm": decision.lens.focal_length_mm,
                          "aperture": decision.lens.aperture_hint},
            "movement":  {"mode": decision.movement.mode},
            "duration":  {"final_s": decision.duration.final_seconds},
            "lighting":  {"mode": decision.lighting.mode,
                          "kelvin": decision.lighting.kelvin_hint},
            "special_event": {
                "enabled": decision.special_event.enabled,
                "delay_s": decision.special_event.delay_seconds,
                "mm":      decision.special_event.magnitude_mm,
            } if decision.special_event.enabled else None,
        },
    }
