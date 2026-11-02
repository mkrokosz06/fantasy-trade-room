import { createContext, useContext, type ReactNode } from "react";

/** Which projection engine a page uses: v1 (usage model) or ml (trained model). */
export type Engine = "v1" | "ml";

const EngineCtx = createContext<{ engine: Engine; base: string }>({ engine: "v1", base: "" });

export function EngineProvider({ engine, children }: { engine: Engine; children: ReactNode }) {
  return <EngineCtx.Provider value={{ engine, base: engine === "ml" ? "/model" : "" }}>{children}</EngineCtx.Provider>;
}

export function useEngine() {
  const { engine, base } = useContext(EngineCtx);
  /** Add ?engine=ml to an API path when running on the model. */
  const ep = (path: string) => (engine === "v1" ? path : `${path}${path.includes("?") ? "&" : "?"}engine=${engine}`);
  return { engine, base, ep, isModel: engine === "ml" };
}
