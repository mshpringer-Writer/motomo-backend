from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

FilterName = Literal["WITNESS", "PRISM", "DISSOLVE", "CURRENT", "SIGNAL"]

@dataclass(frozen=True)
class NSV:
    E: float; C: float; R: float; P: float

@dataclass(frozen=True)
class DeltaNSV:
    dE: float; dC: float; dR: float; dP: float
    @classmethod
    def from_ab(cls, a: NSV, b: NSV) -> "DeltaNSV":
        return cls(dE=b.E-a.E, dC=b.C-a.C, dR=b.R-a.R, dP=b.P-a.P)

@dataclass(frozen=True)
class DSV:
    name: FilterName
    EP: float; TM: float; SO: float; DP: float; DS: float; LE: float; VI: float = 0.0
    anchor_tokens: List[str] = field(default_factory=list)
    supports_special_event: bool = False
    special_event_mode: Optional[str] = None

@dataclass(frozen=True)
class PushDecision:
    mode: Literal["push_in", "pull_back", "locked"]
    intensity: float; description: str

@dataclass(frozen=True)
class MovementDecision:
    mode: Literal["continue", "hold", "arrest", "cut_redirect", "lag_follow"]
    intensity: float; cut_delta: int; description: str

@dataclass(frozen=True)
class DurationDecision:
    base_seconds: float; final_seconds: float; hold_extension: float; description: str

@dataclass(frozen=True)
class FramingDecision:
    angle: Literal["low-angle", "eye-level", "slight-high-angle"]
    dominance: Literal["subject-fills-frame","centered-environment-present","subject-reduced-in-frame"]
    horizon: Literal["stable-horizon", "horizon-drifting", "horizon-stabilizing"]
    composition: Literal["observational","geometric-centered","immersive-soft","kinetic-forward","threshold-held"]
    description: str

@dataclass(frozen=True)
class LensDecision:
    focal_length_mm: int; aperture_hint: str
    lens_family: Literal["wide", "normal", "portrait"]; description: str

@dataclass(frozen=True)
class LightingDecision:
    mode: Literal["available","motivated","directional","warm-organic","revelation"]
    contrast_shift: float; kelvin_hint: int; description: str

@dataclass(frozen=True)
class DepthDecision:
    mode: Literal["deep-focus","moderate-separation","shallow-focus"]; description: str

@dataclass(frozen=True)
class SubjectivityDecision:
    baseline: float; final_position: float; description: str

@dataclass(frozen=True)
class SpecialEventDecision:
    enabled: bool; description: str = ""
    trigger: Optional[str] = None
    delay_seconds: Optional[float] = None
    magnitude_mm: Optional[float] = None

@dataclass(frozen=True)
class CinematicDecision:
    filter_name: FilterName; delta_nsv: DeltaNSV
    push: PushDecision; movement: MovementDecision; duration: DurationDecision
    framing: FramingDecision; lens: LensDecision; lighting: LightingDecision
    depth: DepthDecision; subjectivity: SubjectivityDecision
    special_event: SpecialEventDecision; anchor_tokens: List[str]

FILTERS: Dict[str, DSV] = {
    "WITNESS": DSV("WITNESS", EP=-0.85,TM=-0.90,SO=-0.80,DP=-0.65,DS=2.40,LE=-0.85,VI=0.12,
        anchor_tokens=["locked-off","observational-distance","deep-focus","available-light-only","no-camera-intervention"]),
    "PRISM": DSV("PRISM", EP=-0.30,TM=-0.95,SO=-0.60,DP=-0.90,DS=2.00,LE=0.35,VI=0.08,
        anchor_tokens=["centered-symmetry","geometric-frame","deep-focus","cold-directional-light","formal-stillness"]),
    "DISSOLVE": DSV("DISSOLVE", EP=0.35,TM=-0.40,SO=0.70,DP=0.60,DS=1.80,LE=0.70,VI=0.25,
        anchor_tokens=["warm-imperfect-light","subjective-nearness","shallow-focus","slow-tentative-approach","temporal-softness"]),
    "CURRENT": DSV("CURRENT", EP=0.55,TM=0.90,SO=0.35,DP=0.35,DS=0.60,LE=0.50,VI=0.20,
        anchor_tokens=["tracking-with-subject","kinetic-forward-pressure","motivated-shadows","medium-shallow-separation","fast-precise-energy"]),
    "SIGNAL": DSV("SIGNAL", EP=-0.10,TM=-0.70,SO=-0.20,DP=0.50,DS=1.60,LE=0.60,VI=0.10,
        anchor_tokens=["held-threshold-frame","subject-isolation","revelation-light","delayed-reframe","just-outside-perspective"],
        supports_special_event=True, special_event_mode="mutation"),
}

class CinematicEngine:
    def build_decision(self, nsv_a, nsv_b, dsv, base_duration_seconds=10.0):
        dn = DeltaNSV.from_ab(nsv_a, nsv_b)
        return CinematicDecision(
            filter_name=dsv.name, delta_nsv=dn,
            push=self.compute_push(dsv,dn),
            movement=self.compute_movement(dsv,dn),
            duration=self.compute_duration_phase1(dsv,dn,base_duration_seconds),
            framing=self.compute_control_framing(dsv,dn,nsv_b),
            lens=self.compute_lens(dsv,dn,nsv_b),
            lighting=self.compute_lighting(dsv,dn),
            depth=self.compute_depth(dsv,dn),
            subjectivity=self.compute_subjectivity(dsv,dn),
            special_event=self.compute_special_event(dsv,dn,base_duration_seconds),
            anchor_tokens=dsv.anchor_tokens,
        )

    def compute_push(self, dsv, dn):
        dp = max(0.0, min(1.0, dn.dP))
        if abs(dsv.EP) < 0.05:
            return PushDecision("locked", 0.0, "neutral proximity")
        if dsv.EP > 0:
            intensity = max(0.0, dsv.EP*(1.0-dp))
            if intensity > 0.12:
                return PushDecision("push_in", round(intensity,3), "approach weakens under price")
            return PushDecision("locked", 0.0, "approach arrested by price")
        retreat = min(1.0, abs(dsv.EP)*(1.0+dp*0.6))
        return PushDecision("pull_back", round(retreat,3), "retreat intensifies under price")

    def compute_movement(self, dsv, dn):
        signal = dsv.TM * dn.dR
        if dsv.name == "CURRENT" and dn.dC < -0.20:
            return MovementDecision("lag_follow", min(1.0,abs(signal)), 1, "camera lags the subject as control drops")
        if signal > 0.50:
            return MovementDecision("cut_redirect", min(1.0,signal), 2, "risk drives motion interruption and redirect")
        if signal > 0.25:
            return MovementDecision("arrest", min(1.0,signal), 1, "motion briefly arrests, then resumes")
        if signal < -0.25:
            return MovementDecision("hold", min(1.0,abs(signal)), 0, "stillness deepens under risk")
        return MovementDecision("continue", min(1.0,abs(signal)), 0, "movement remains close to prior state")

    def compute_duration_phase1(self, dsv, dn, base):
        ds_norm = max(0.0, min(1.0,(dsv.DS-0.4)/2.6))
        factor = 1.0 + ds_norm*max(0.0,dn.dP)*0.7
        final = round(base*factor,1)
        hold = round(ds_norm*max(0.0,dn.dP)*3.0,1)
        return DurationDecision(base, final, hold, "phase-1 DS×dP scaling")

    def compute_control_framing(self, dsv, dn, nsv_b):
        c, dc = nsv_b.C, dn.dC
        angle = "low-angle" if c>0.40 else ("eye-level" if c>-0.10 else "slight-high-angle")
        if dsv.name == "WITNESS":
            comp="observational"; dom="centered-environment-present" if c>-0.15 else "subject-reduced-in-frame"; hor="stable-horizon"
        elif dsv.name == "PRISM":
            comp="geometric-centered"; dom="subject-fills-frame" if c>0.35 else "subject-reduced-in-frame"; hor="stable-horizon"
        elif dsv.name == "DISSOLVE":
            comp="immersive-soft"; dom="centered-environment-present"; hor="horizon-drifting" if dc<-0.20 else "stable-horizon"
        elif dsv.name == "CURRENT":
            comp="kinetic-forward"; dom="subject-fills-frame" if c>0.15 else "centered-environment-present"; hor="horizon-drifting" if dc<-0.20 else "stable-horizon"
        else:
            comp="threshold-held"; dom="centered-environment-present" if c>-0.15 else "subject-reduced-in-frame"; hor="stable-horizon"
        return FramingDecision(angle, dom, hor, comp, f"{dsv.name}: C={c:+.2f} dC={dc:+.2f}")

    def compute_lens(self, dsv, dn, nsv_b):
        r = nsv_b.R
        if dsv.name == "WITNESS":   return LensDecision(35,"stopped-down","wide","observational wide, world present")
        if dsv.name == "PRISM":     return LensDecision(28,"stopped-down","wide","geometric deep-focus wide")
        if dsv.name == "DISSOLVE":  return LensDecision(85 if dsv.DP>0.35 or r>0.65 else 50,"f1.8","portrait","warm shallow portrait")
        if dsv.name == "CURRENT":   return LensDecision(40 if r<0.70 else 50,"f2.8","normal","kinetic normal lens")
        return LensDecision(75 if r>0.60 else 50,"f2.0","portrait","subject separated from softening world")

    def compute_lighting(self, dsv, dn):
        dp = max(0.0, dn.dP)
        if dsv.name == "WITNESS":   return LightingDecision("available",0.0,4500,"available light, no shaping")
        if dsv.name == "PRISM":     return LightingDecision("directional",round(0.25+dp*0.2,3),4800,"cold directional")
        if dsv.name == "DISSOLVE":  return LightingDecision("warm-organic",round(0.10+dp*0.15,3),3800,"warm imperfect practical")
        if dsv.name == "CURRENT":   return LightingDecision("motivated",round(0.20+dp*0.2,3),4300,"motivated contrast")
        return LightingDecision("revelation",round(0.30+dp*0.25,3),5000,"revelation directional")

    def compute_depth(self, dsv, dn):
        s = dsv.DP * dn.dR
        if s > 0.20: return DepthDecision("shallow-focus","subject isolated")
        if s < -0.20: return DepthDecision("deep-focus","environment active")
        return DepthDecision("moderate-separation","subject clear, world implied")

    def compute_subjectivity(self, dsv, dn):
        b = dsv.SO
        if abs(b) < 0.05:
            return SubjectivityDecision(0.0,0.0,"neutral observer")
        if b > 0:
            raw = b + dn.dE*b
            final = max(0.0, raw)
            return SubjectivityDecision(round(b,3),round(final,3),"inside filter: stays inside, may soften")
        else:
            mag = abs(dn.dE)*abs(b)
            raw = b - mag if dn.dE<0 else b + mag*0.5
            final = min(0.0, raw)
            return SubjectivityDecision(round(b,3),round(final,3),"outside filter: stays outside, never crosses zero")

    def compute_special_event(self, dsv, dn, base):
        if not dsv.supports_special_event: return SpecialEventDecision(enabled=False)
        if max(dn.dP, dn.dR) < 0.35: return SpecialEventDecision(enabled=False)
        delay = round(base*(0.25+max(0.0,dn.dP)*0.15),1)
        mag = round(3.0*(1+max(0.0,dn.dP)*0.3),1)
        return SpecialEventDecision(True,"threshold reframe","revelation-or-price-spike",delay,mag)

class PromptCompiler:
    def compile_human(self, d: CinematicDecision) -> str:
        dn = d.delta_nsv
        lines = [
            f"[{d.filter_name}] ΔE={dn.dE:+.2f} ΔC={dn.dC:+.2f} ΔR={dn.dR:+.2f} ΔP={dn.dP:+.2f}",
            f"Push: {d.push.description}",
            f"Movement: {d.movement.description}",
            f"Framing: {d.framing.description}",
            f"Lens: {d.lens.description}",
            f"Lighting: {d.lighting.description}",
            f"Depth: {d.depth.description}",
            f"Subjectivity: {d.subjectivity.description} (baseline={d.subjectivity.baseline:+.2f} → {d.subjectivity.final_position:+.2f})",
        ]
        if d.special_event.enabled:
            lines.append(f"Event: {d.special_event.description} @ {d.special_event.delay_seconds}s, {d.special_event.magnitude_mm}mm")
        return "\n".join(lines)

    def compile_render_tokens(self, d: CinematicDecision) -> List[str]:
        tokens = list(d.anchor_tokens)
        tokens += [d.framing.angle, d.framing.dominance, d.framing.horizon, d.framing.composition]
        tokens += [f"{d.lens.focal_length_mm}mm", d.lens.aperture_hint, d.depth.mode]
        push_map = {"push_in":"slow-push-in","pull_back":"slow-pull-back","locked":"no-approach-change"}
        tokens.append(push_map[d.push.mode])
        move_map = {"continue":"movement-continues","hold":"held-frame","arrest":"brief-motion-arrest","cut_redirect":"hard-cut-redirect","lag_follow":"camera-lags-subject"}
        tokens.append(move_map[d.movement.mode])
        tokens += [f"{d.duration.final_seconds:.0f}s-shot"]
        if d.duration.hold_extension > 0: tokens.append(f"hold-{d.duration.hold_extension:.1f}s")
        light_map = {"available":"available-light-only","motivated":"motivated-contrast","directional":"cold-directional","warm-organic":"warm-organic-practicals","revelation":"revelation-light"}
        tokens.append(light_map[d.lighting.mode])
        if d.subjectivity.final_position > 0.45: tokens.append("strongly-inside")
        elif d.subjectivity.final_position > 0.0: tokens.append("semi-subjective")
        elif d.subjectivity.final_position < -0.45: tokens.append("cold-observer")
        else: tokens.append("just-outside")
        if d.special_event.enabled:
            tokens += ["delayed-reframe", f"{d.special_event.magnitude_mm:.1f}mm-reframe", f"reframe-at-{d.special_event.delay_seconds:.1f}s"]
        seen = set(); out = []
        for t in tokens:
            if t not in seen: out.append(t); seen.add(t)
        return out

    def compile_render_prompt(self, d: CinematicDecision) -> str:
        return ", ".join(self.compile_render_tokens(d))

if __name__ == "__main__":
    engine = CinematicEngine()
    compiler = PromptCompiler()
    A = NSV(E=0.55, C=0.35, R=0.45, P=0.25)
    B = NSV(E=0.30, C=0.10, R=0.85, P=0.88)
    DURS = {"WITNESS":16.0,"PRISM":12.0,"DISSOLVE":13.0,"CURRENT":6.0,"SIGNAL":14.0}

    all_tokens = {}
    for name, dsv in FILTERS.items():
        d = engine.build_decision(A, B, dsv, DURS[name])
        tokens = compiler.compile_render_tokens(d)
        all_tokens[name] = set(tokens)
        print(f"\n{'═'*65}")
        print(f"  {name}")
        print(f"{'─'*65}")
        print(compiler.compile_human(d))
        print(f"\n  RENDER: {compiler.compile_render_prompt(d)}")

    # Overlap analysis
    print(f"\n{'═'*65}")
    print("  TOKEN OVERLAP MATRIX")
    print(f"{'─'*65}")
    names = list(FILTERS.keys())
    for i, a in enumerate(names):
        for b in names[i+1:]:
            shared = all_tokens[a] & all_tokens[b]
            pct = len(shared)/min(len(all_tokens[a]),len(all_tokens[b]))*100
            print(f"  {a:10} ∩ {b:10} = {len(shared):2d} shared ({pct:.0f}%)  {sorted(shared)[:3]}...")
