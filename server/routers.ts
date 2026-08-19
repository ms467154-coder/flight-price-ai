import { z } from "zod";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { createPrediction, getPredictionsByUser } from "./db";
import { getModelManifest, runFlightInference } from "./ml/model";

const flightInput = z.object({
  airline: z.string().min(1).max(64),
  flight: z.string().min(1).max(32),
  source_city: z.string().min(1).max(64),
  destination_city: z.string().min(1).max(64),
  departure_time: z.string().min(1).max(32),
  arrival_time: z.string().min(1).max(32),
  stops: z.string().min(1).max(32),
  class: z.string().min(1).max(32),
  duration: z.coerce.number().finite().nonnegative(),
  days_left: z.coerce.number().int().nonnegative(),
});

export const appRouter = router({
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(async ({ ctx }) => { ctx.res.clearCookie(COOKIE_NAME, { ...getSessionCookieOptions(ctx.req), maxAge: -1 }); return { success: true } as const; }),
  }),
  model: router({
    manifest: publicProcedure.query(() => getModelManifest()),
  }),
  predictions: router({
    history: protectedProcedure.query(({ ctx }) => getPredictionsByUser(ctx.user.id)),
    create: protectedProcedure.input(flightInput).mutation(async ({ ctx, input }) => {
      const result = await runFlightInference(input);
      const model = result.model!;
      const saved = await createPrediction({ userId: ctx.user.id, inputs: JSON.stringify(input), predictedPrice: String(result.predictedPrice), modelId: model.model_id });
      return { id: saved.id, predictedPrice: result.predictedPrice, model, inputs: input, createdAt: new Date() };
    }),
  }),
});

export type AppRouter = typeof appRouter;
