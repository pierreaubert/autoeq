(* autoeq-qa case autoeq-qa.re16-spatial-quadrature.v1 (catalogue family RE16).
   Continuous-area quadrature and risk, derived independently: the
   analytic Gaussian-field area integral (erf closed form) versus an
   explicit midpoint quadrature sum; the Abramowitz-Stegun erf and
   Acklam inverse-normal rational approximations checked against the
   engine Erf/InverseErf; exact discrete expected value and fractional
   tail CVaR (zero-mass points carry no mass but do not end the tail). *)

Off[General::munfl];
a = 5.; cx = 1.; cy = 0.5; s = 0.4; b = 20.;
lx = 2.; ly = 1.;
ix = N[s Sqrt[2 Pi]/2 (Erf[(lx - cx)/(s Sqrt[2])] - Erf[(0 - cx)/(s Sqrt[2])]), 16];
iy = N[s Sqrt[2 Pi]/2 (Erf[(ly - cy)/(s Sqrt[2])] - Erf[(0 - cy)/(s Sqrt[2])]), 16];
analytic = N[a ix iy + b lx ly, 16];
nx = 8; ny = 6;
w = N[lx ly/(nx ny), 16];
field[x_, y_] := a Exp[-((x - cx)^2 + (y - cy)^2)/(2 s^2)] + b;
qsum = N[w Sum[field[(i - 0.5) lx/nx, (j - 0.5) ly/ny], {i, nx}, {j, ny}], 16];
constQuad = N[w Sum[b, {i, nx}, {j, ny}], 16];
cdfX = {-2., -1., -0.5, 0., 0.5, 1., 2.};
cdfRef = N[0.5 (1 + Erf[#/Sqrt[2]]) & /@ cdfX, 16];
invP = {0.001, 0.025, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.999};
invRef = N[Sqrt[2] InverseErf[2 # - 1] & /@ invP, 16];
losses = {1.5, 0.5, 3.0, 2.0, 99.0};
weights = {0.1, 0.4, 0.3, 0.2, 0.0};
expected = N[losses . weights, 16];
cvar[alpha_] := Module[{pairs, mass = 0., acc = 0., take},
   pairs = Reverse[SortBy[Transpose[{losses, weights}], First]];
   Do[
    If[pair[[2]] <= 0., Continue[]];
    take = Min[alpha - mass, pair[[2]]];
    If[take <= 0., Break[]];
    acc += take pair[[1]];
    mass += take;
    If[mass >= alpha, Break[]],
    {pair, pairs}];
   N[acc/mass, 16]];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re16-spatial-quadrature.v1",
  "family" -> "RE16",
  "sample_rate_hz" -> 48000.,
  "grid" -> "analytic rectangle [0,2]x[0,1]; explicit 8x6 midpoint quadrature",
  "units" -> <|"field" -> "dB loss", "area" -> "m^2", "probability" -> "fraction"|>,
  "calibration" -> "synthetic",
  "field" -> <|"a" -> a, "cx" -> cx, "cy" -> cy, "s" -> s, "b" -> b|>,
  "analytic_integral" -> analytic,
  "quadrature_integral" -> qsum,
  "quadrature_cells" -> {nx, ny},
  "const_field_quadrature" -> constQuad,
  "const_field_analytic" -> N[b lx ly, 16],
  "cdf_points" -> cdfX,
  "cdf_reference" -> cdfRef,
  "inverse_points" -> invP,
  "inverse_reference" -> invRef,
  "losses" -> losses,
  "loss_weights" -> weights,
  "expected_loss" -> expected,
  "cvar_alpha_0p5" -> cvar[0.5],
  "cvar_alpha_0p9" -> cvar[0.9]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
