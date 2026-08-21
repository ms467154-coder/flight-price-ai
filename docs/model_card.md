# Model Card

## Intended use

This release estimates a historical fare-like target from route, airline, timing, cabin, duration, stops, and booking-horizon features. It is intended for exploratory planning and product demonstrations.

## Limitations

The model is not a live fare feed and does not account for inventory changes, promotions, holidays, operational disruption, or future market shocks. Predictions are estimates, not guaranteed future prices or booking quotes.

## Release evidence

Metrics, feature contract, limitations, release status, and model identity are stored in `ml_artifacts/model_manifest.json` and surfaced live by the application.
