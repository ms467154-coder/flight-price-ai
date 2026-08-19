import { describe, expect, it } from "vitest";
import { createPrediction, getDb, getPredictionsByUser } from "./db";

describe("prediction persistence contract", () => {
  it.skipIf(!process.env.DATABASE_URL)("persists serialized inputs and returns newest-first history", async () => {
    const db = await getDb();
    expect(db).toBeTruthy();
    const userId = 1;
    const older = await createPrediction({ userId, inputs: JSON.stringify({ airline: "Vistara", days_left: 20 }), predictedPrice: "12000", modelId: "test-model" });
    await new Promise(resolve => setTimeout(resolve, 10));
    const newer = await createPrediction({ userId, inputs: JSON.stringify({ airline: "Indigo", days_left: 5 }), predictedPrice: "8000", modelId: "test-model" });
    const rows = await getPredictionsByUser(userId);
    const createdIds = rows.filter(row => row.id === older.id || row.id === newer.id);
    expect(createdIds[0]?.id).toBe(newer.id);
    expect(createdIds[0]?.userId).toBe(userId);
    expect(createdIds[0]?.predictedPrice).toBe("8000");
    expect(createdIds[0]?.modelId).toBe("test-model");
    expect(createdIds[0]?.createdAt).toBeInstanceOf(Date);
    expect(JSON.parse(createdIds[0]?.inputs || "{}")).toEqual({ airline: "Indigo", days_left: 5 });
  });
});
