# Roadmap

This roadmap describes the planned direction of wellcast. Priorities can change
with user feedback; please share yours in
[GitHub Discussions](https://github.com/olaflaitinen/wellcast/discussions).

## 0.2.0: Nodal analysis

- Vertical lift performance (VLP): multiphase flow correlations for tubing
  (for example Hagedorn-Brown and Beggs-Brill).
- Nodal analysis: operating point from IPR and VLP, sensitivity to tubing size,
  wellhead pressure and gas lift.
- More PVT correlations (for example Glaso, Al-Marhoun, Petrosky-Farshad) and
  gas viscosity (Lee-Gonzalez-Eakin).

## 0.3.0: Rate transient analysis

- Rate transient analysis: normalized rate, material balance time and
  diagnostic plots.
- Type curves (Fetkovich, Blasingame) with matching tools.
- Modified hyperbolic and Duong decline models for unconventional wells.

## 1.0.0: Stable API

- Stable, documented public API with a deprecation policy.
- Performance work for large fields (thousands of wells).
- Extended validation against published field cases.
