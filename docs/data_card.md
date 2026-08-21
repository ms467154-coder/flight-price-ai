# Data Card

## Schema

The model consumes ten fields: airline, flight, source city, departure time, stops, arrival time, destination city, class, duration, and days left. The canonical contract is stored in the release artifacts and enforced by the ML validation code.

## Quality policy

Training and release validation check schema compatibility, missingness, duplicates, domain constraints, category values, and target quality. The application validates request types at the tRPC boundary before invoking Python inference.

## Known limitations

The training dataset represents historical observations and should not be interpreted as a complete real-time market. New deployments should compare incoming distributions against the training reference before promoting a new model.
