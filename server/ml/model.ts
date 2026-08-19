import { readFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";

export type ModelManifest = {
  model_id: string;
  model_name: string;
  model_type: string;
  artifact: string;
  features: string[];
  target: string;
  cv_metrics: Record<string, number>;
  test_metrics: Record<string, number>;
  limitations: string[];
  intended_use: string;
  release_status: string;
  created_at: string;
  [key: string]: unknown;
};

const projectRoot = process.env.APP_ROOT || process.cwd();
const manifestPath = path.resolve(projectRoot, "ml_artifacts/model_manifest.json");

export async function getModelManifest(): Promise<ModelManifest> {
  return JSON.parse(await readFile(manifestPath, "utf8")) as ModelManifest;
}

export async function runFlightInference(input: Record<string, unknown>): Promise<{ predictedPrice: number; model: ModelManifest }> {
  const script = path.resolve(projectRoot, "server/ml/infer.py");
  const python = process.env.PYTHON_BIN || "python3";
  const output = await new Promise<string>((resolve, reject) => {
    const child = spawn(python, [script], { cwd: projectRoot, env: { ...process.env, PYTHONUNBUFFERED: "1" }, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => { child.kill("SIGTERM"); reject(new Error("Model inference timed out")); }, 30_000);
    child.stdout.on("data", chunk => { stdout += chunk.toString(); });
    child.stderr.on("data", chunk => { stderr += chunk.toString(); });
    child.on("error", error => { clearTimeout(timer); reject(error); });
    child.on("close", code => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error(stderr.trim() || stdout.trim() || `Inference exited with ${code}`));
      else resolve(stdout.trim());
    });
    child.stdin.end(JSON.stringify(input));
  });
  const result = JSON.parse(output) as { predictedPrice?: number; model?: ModelManifest; error?: string };
  if (result.error || typeof result.predictedPrice !== "number" || !result.model) throw new Error(result.error || "Invalid model response");
  return { predictedPrice: result.predictedPrice, model: result.model };
}
