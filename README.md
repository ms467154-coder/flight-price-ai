# Flight Price AI

> A transparent flight-fare estimation workspace powered by a validated regression model and a production-oriented ML/MLOps lifecycle.

[![CI](https://github.com/ms467154-coder/flight-price-ai/actions/workflows/ml-ci.yml/badge.svg)](https://github.com/ms467154-coder/flight-price-ai/actions/workflows/ml-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-C0392B.svg)](LICENSE)

Flight Price AI helps travelers explore an estimated fare before booking by combining a responsive React application, authenticated prediction workflows, a MySQL history layer, and a release-manifest-backed Python inference service. The interface is intentionally transparent: model identity, evaluation metrics, feature contract, release status, and limitations are loaded from the committed model manifest rather than copied into the UI.

## Product overview

The application has two connected experiences. Public visitors can explore the model, review the methodology, and read release notes. Authenticated users can submit route and booking-context features, receive an estimate, and revisit their newest-first prediction history with the complete serialized input snapshot and model release used.

> **Important:** A prediction is an estimate derived from historical training data and model assumptions. It is not a guaranteed future price, a live market quote, or a booking commitment.

| Capability | Implementation |
| --- | --- |
| Public landing page | React 19, Tailwind CSS 4, cream/off-white and muted-red visual system |
| Authenticated predictions | Manus OAuth with protected tRPC mutation |
| ML inference | Node.js child process invoking Python, joblib model artifact, live release manifest |
| Persistence | MySQL/TiDB through Drizzle ORM with user foreign key and newest-first ordering |
| Model transparency | `ml_artifacts/model_manifest.json` exposed through a typed public procedure |
| Release documentation | Footer-linked `/release-notes` page describing the ML CI workflow |
| Production runtime | Custom Dockerfile with Node.js and Python 3 runtime |

## Architecture

```text
React client
    │  tRPC / React Query
    ▼
Express + tRPC server ─── Manus OAuth session
    │
    ├── public model.manifest
    ├── protected predictions.create
    └── protected predictions.history
          │
          ├── MySQL/TiDB predictions table
          └── Python inference bridge
                │
                ├── ml_artifacts/model_manifest.json
                └── ml_artifacts/models/flight_price_model.joblib
```

The prediction contract follows the model’s ten features: airline, flight, source city, departure time, stops, arrival time, destination city, class, duration, and days left. Zod validates the request at the tRPC boundary before the Python process is spawned.

## Model transparency

The current release is a `HistGradientBoostingRegressor` trained and validated through the repository’s ML pipeline. The application reads the following values live from the release manifest:

| Metadata | Source |
| --- | --- |
| Model ID and release status | `ml_artifacts/model_manifest.json` |
| Test R², MAE, RMSE, and related metrics | `test_metrics` in the manifest |
| Feature list and target | `features` and `target` in the manifest |
| Model limitations | `limitations` in the manifest |
| Artifact identity | `checksums.json` and model manifest |

The repository also retains data validation reports, evaluation reports, error analysis, feature schema, model comparison output, and checksum metadata so that a model release can be inspected independently of the UI.

## Repository structure

```text
client/                 React application and branded product UI
drizzle/                MySQL schema and migrations
ml/                     Training, validation, drift, and release utilities
ml_artifacts/           Validated model, manifest, metrics, and checksums
server/                 tRPC procedures, database helpers, Python bridge
shared/                 Shared application constants and types
docs/                   Model, data, monitoring, and release documentation
.github/workflows/      ML validation and release CI
Dockerfile              Node + Python production runtime
requirements-ml.txt     Python runtime dependencies
```

## Local development

### Prerequisites

Node.js 22+, pnpm 10+, Python 3.11+, and a MySQL-compatible database are recommended. Python dependencies are installed from `requirements-ml.txt`; JavaScript dependencies are installed from the committed `pnpm-lock.yaml`.

### Install and run

```bash
pnpm install
python3 -m pip install -r requirements-ml.txt
pnpm dev
```

The development server reads application secrets from the environment. Do not commit `.env` files or replace the platform-provided authentication and database variables with literal values.

### Validate the project

```bash
pnpm check
pnpm test -- --run
pnpm build
```

The test suite covers the authentication cookie contract, live manifest loading, Python inference compatibility, and the prediction persistence contract. The database integration test runs when `DATABASE_URL` is available and verifies serialized inputs, predicted price, model ID, created timestamp, user ownership, and newest-first ordering.

## Production build

The repository includes a custom Dockerfile because the application must execute Python inference in the same runtime as the Node.js server.

```bash
docker build -t flight-price-ai .
docker run --env-file .env -p 3000:3000 flight-price-ai
```

The container installs Node dependencies, Python ML dependencies, builds the Vite client and Express server, and starts `dist/index.js`. The server uses the platform-provided `PORT` value and reads model files from the application root.

## ML lifecycle

The ML pipeline is designed to make data and model changes reviewable. Data validation checks the canonical schema, missingness, duplicates, domain constraints, category values, and target quality. Training fingerprints the dataset, records the feature contract and package versions, compares baseline and candidate models, evaluates held-out performance, writes error analysis, and emits a checksum-backed release bundle. The GitHub Actions workflow repeats the validation, training, testing, and release verification sequence on pull requests and pushes to the default branch.

Future operational work can add scheduled drift monitoring, realized-price feedback, deployment promotion automation, and a model registry. Those integrations should remain separate from the current validated release artifacts rather than being implied by the UI.

## Security and governance

Authentication is handled through the existing Manus OAuth integration. Prediction creation and history are protected by the server-side `protectedProcedure`, and history queries are scoped by the authenticated user ID. Model metadata is served from the repository release bundle; secrets remain environment-managed. The application clearly presents the estimate disclaimer to reduce the risk of interpreting the output as a guaranteed price.

## Contributing

Keep changes focused and reviewable. For ML changes, update the relevant data card, model card, pipeline documentation, tests, and release artifacts together. For application changes, preserve the tRPC contract, validate external input with Zod, and include loading, error, and authenticated-state behavior. Run `pnpm check`, `pnpm test -- --run`, and `pnpm build` before opening a pull request.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Maintainer

**Mohamed Salem** — Flight Price AI
