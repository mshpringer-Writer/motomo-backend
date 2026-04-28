"""
Integration test: engine SSV/NSV → compiler → render_lines
Simulates what /compute-override returns after the patch.
"""
import importlib.util, pathlib, sys

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

# Point compiler at patched v0.3.1
import shutil
shutil.copy(
    "/home/claude/motomo_behavioral_compiler_v03_patched.py",
    "/home/claude/motomo_behavioral_compiler.py"
)

engine = load("/home/claude/motomo_engine.py", "engine")
patch  = load("/home/claude/motomo_integration_patch.py", "patch")

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: print(f"  ✓  {name}"); passed += 1
    else:    print(f"  ✗  {name}  ← {detail}"); failed += 1

# ── Uri SSV (from engine __main__) ────────────────────────────────────────────
uri_ssv = engine.SSV(
    RT=0.22, PS=0.88, DT=0.35,
    loyalty=0.82, shame=0.88, validation=0.78,
    conflict_avoidance=0.82, Yu="relational_security",
)
base_nsv = engine.NSV(E=0.3, C=0.2, R=0.3, P=0.2, Dc=0.5, Du=0.6)

print("\n── SSV adapter ──────────────────────────────────────────────────")
d = patch.ssv_to_compiler_dict(uri_ssv)
check("R=RT",                 d["R"] == 0.22)
check("P=1-PS (tolerance)",   abs(d["P"] - (1-0.88)) < 0.001, f"got {d['P']}")
check("W=loyalty",            d["W"] == 0.82)
check("A=loyalty",            d["A"] == 0.82)
check("all values in [0,1]",  all(0 <= v <= 1 for v in d.values()))

print("\n── NSV adapter ──────────────────────────────────────────────────")
n = patch.nsv_to_compiler_dict(base_nsv)
check("E normalised to [0,1]",  0 <= n["E"] <= 1, f"got {n['E']}")
check("P_imm = nsv.P",          n["P_imm"] == base_nsv.P)
check("P_lat = nsv.Du",         n["P_lat"] == base_nsv.Du)
check("all values in [0,1]",    all(0 <= v <= 1 for v in n.values()))

print("\n── build_ltx_prompt (flirt, override, gap=0.319) ────────────────")
signal   = engine.ExternalSignal(wife_message_active=True)
nsv_upd  = engine.apply_nsv_update(base_nsv, uri_ssv, signal)
rankings = engine.compute_rankings(uri_ssv, base_nsv, signal)
top      = rankings[0]
rec      = rankings[1]

result = patch.build_ltx_prompt(
    character_name="Uri — The Wounded Loyalist",
    top_action={"action":"flirt_no_commit","label":"Flirt without commitment","score":0.55},
    ssv=uri_ssv, nsv=nsv_upd, signal=signal,
    beat_context="Maya invites him home",
    score_gap=0.319, selection_type="override",
    recommended_action=rec,
)

check("render_lines in response",    "render_lines" in result)
check("behavioral_tokens in response","behavioral_tokens" in result)
check("drivers in response",          "drivers" in result)
check("pressure_band in response",    "pressure_band" in result)
check("render_lines non-empty",       len(result["render_lines"]) > 0)
check("no 'score_gap' in prompt text",
      "score_gap" not in result["compiled_prompt"].split("PERFORMANCE")[1].split("CAMERA")[0],
      "engine metadata leaked into PERFORMANCE block")
check("no 'pressure' in prompt text",
      "pressure=" not in result["compiled_prompt"].split("PERFORMANCE")[1].split("CAMERA")[0])
check("PERFORMANCE block present",   "PERFORMANCE:" in result["compiled_prompt"])

print("\n── SSV1 vs SSV2 via patched prompt ──────────────────────────────")
# SSV2 — low loyalty, low shame
ssv2 = engine.SSV(RT=0.91, PS=0.15, DT=0.88, loyalty=0.18,
                   shame=0.15, validation=0.46, conflict_avoidance=0.20)
signal2  = engine.ExternalSignal(wife_message_active=False)
nsv2_upd = engine.apply_nsv_update(base_nsv, ssv2, signal2)

r1 = patch.build_ltx_prompt("Uri SSV1","flirt_no_commit" and
     {"action":"flirt_no_commit","label":"Flirt","score":0.55},
     uri_ssv, nsv_upd, signal, "bar", score_gap=0.319)
r2 = patch.build_ltx_prompt("Uri SSV2",
     {"action":"flirt_no_commit","label":"Flirt","score":0.55},
     ssv2, nsv2_upd, signal2, "bar", score_gap=0.032)

check("SSV1 wife_msg=True → HIGH band (correct: wife just messaged)",  r1["pressure_band"] == "HIGH", f"got {r1["pressure_band"]}")
check("SSV2 → LOW band",  r2["pressure_band"] == "LOW", f"got {r2['pressure_band']}")
check("SSV1/SSV2 render_lines differ",
      r1["render_lines"] != r2["render_lines"])

print(f"\n── Sample output (SSV1 flirt override) ─────────────────────────")
print(result["compiled_prompt"])

print(f"\n{'='*55}")
print(f"  {passed} passed   {failed} failed")
print(f"{'='*55}")
sys.exit(0 if failed == 0 else 1)
