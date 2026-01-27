Heat Flux Extrapolation (Cases 45-47): EXCELLENT 
┌──────┬───────────┬────────────────┬────────┬───────────┬────────┐
│ Case │ Heat Flux │ Above Training │  MAE   │ Max Error │  RMSE  │
├──────┼───────────┼────────────────┼────────┼───────────┼────────┤
│ 0045 │ 280k W/m² │ +12%           │ 0.30 K │ 1.27 K    │ 0.38 K │
├──────┼───────────┼────────────────┼────────┼───────────┼────────┤
│ 0046 │ 310k W/m² │ +24%           │ 0.35 K │ 1.64 K    │ 0.46 K │
├──────┼───────────┼────────────────┼────────┼───────────┼────────┤
│ 0047 │ 340k W/m² │ +36%           │ 0.48 K │ 2.10 K    │ 0.61 K │
└──────┴───────────┴────────────────┴────────┴───────────┴────────┘
Training range: 50k - 250k W/m²

Analysis:
- Sub-degree errors even at 36% extrapolation!
- Physics-informed loss is doing its job beautifully
- Heat flux parameter learned very well

T_initial Extrapolation (Cases 48-49): POOR 
┌──────┬───────────────┬──────────────────┬─────────┬───────────┬─────────┐
│ Case │   T_initial   │ Outside Training │   MAE   │ Max Error │  RMSE   │
├──────┼───────────────┼──────────────────┼─────────┼───────────┼─────────┤
│ 0048 │ 250 K (-23°C) │ -23 K below      │ 19.91 K │ 23.15 K   │ 19.96 K │
├──────┼───────────────┼──────────────────┼─────────┼───────────┼─────────┤
│ 0049 │ 383 K (110°C) │ +20 K above      │ 17.10 K │ 20.01 K   │ 17.13 K │
└──────┴───────────────┴──────────────────┴─────────┴───────────┴─────────┘
Training range: 273 - 363 K (0°C - 90°C)

Analysis:
- Initial condition extrapolation is much harder
- Model hasn't seen these temperature ranges
- 17-20K errors are significant but not catastrophic

Key Insights:

What worked:
- Heat flux extrapolation is excellent
- PINN generalization for boundary condition variations
- Physics loss helps the model behave correctly outside training

What struggled:
- Initial temperature extrapolation is weak
- The model memorized the temperature range rather than learning the physics fully

Why this pattern?

Heat flux affects how heat enters (boundary condition) - the physics equation directly
models this, so the PINN extrapolates well.

T_initial affects where you start (initial condition) - this shifts the entire
solution space, harder for the network to extrapolate.