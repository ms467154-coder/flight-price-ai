import { desc, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertPrediction, InsertUser, predictions, users } from "../drizzle/schema";
import { ENV } from "./_core/env";

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) _db = drizzle(process.env.DATABASE_URL);
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  const db = await getDb();
  if (!db) return;
  const values: InsertUser = { ...user, role: user.openId === ENV.ownerOpenId ? "admin" : user.role ?? "user" };
  await db.insert(users).values(values).onDuplicateKeyUpdate({ set: { name: values.name, email: values.email, lastSignedIn: new Date() } });
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result[0];
}

export async function createPrediction(prediction: InsertPrediction) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available");
  const result = await db.insert(predictions).values(prediction);
  return { id: Number(result[0].insertId) };
}

export async function getPredictionsByUser(userId: number) {
  const db = await getDb();
  if (!db) throw new Error("Database is not available");
  return db.select().from(predictions).where(eq(predictions.userId, userId)).orderBy(desc(predictions.createdAt)).limit(100);
}
