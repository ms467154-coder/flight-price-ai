# Monitoring

Offline monitoring utilities compare training and production distributions for numerical and categorical features. A production integration should schedule these checks, retain the reports, alert on threshold breaches, and compare realized prices with predictions when outcome data becomes available.

The application currently exposes model metadata and limitations, while operational alerting and scheduled execution remain deployment work.

# Release Process

Model changes move through candidate, validated, staging, production, and archived states. A candidate must pass data validation, tests, evaluation gates, manifest checks, and checksum verification before it is considered validated. Promotion to staging or production should be an explicit operational action with rollback to the previous artifact bundle.
