import { describe, expect, it } from "vitest";
import { getModelManifest, runFlightInference } from "./ml/model";

describe("flight prediction ML integration", () => {
  const input = {
    airline: "Vistara",
    flight: "UK-880",
    source_city: "Delhi",
    destination_city: "Mumbai",
    departure_time: "Evening",
    arrival_time: "Night",
    stops: "one",
    class: "Economy",
    duration: 2.5,
    days_left: 14,
  };

  it("loads live metadata from the release manifest", async () => {
    const manifest = await getModelManifest();
    expect(manifest.model_id).toBeTruthy();
    expect(manifest.features).toEqual(["airline", "flight", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class", "duration", "days_left"]);
    expect(manifest.test_metrics.R2).toBeTypeOf("number");
  });

  it("returns a finite numeric price through the Python bridge", async () => {
    const result = await runFlightInference(input);
    expect(result.predictedPrice).toBeTypeOf("number");
    expect(Number.isFinite(result.predictedPrice)).toBe(true);
    expect(result.model.model_id).toBeTruthy();
  });
});
