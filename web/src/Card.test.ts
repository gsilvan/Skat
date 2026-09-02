import { describe, expect, it } from "vitest";
import { spritePosition } from "./Card";

describe("spritePosition", () => {
  it("maps the Skat engine's suit and rank order to the external sprite", () => {
    expect(spritePosition({ suit: "♣", rank: "A" })).toEqual({ x: "0%", y: "0%" });
    expect(spritePosition({ suit: "♦", rank: "7" })).toEqual({ x: "100%", y: "100%" });
    expect(spritePosition({ suit: "♥", rank: "10" })).toEqual({
      x: `${(4 / 7) * 100}%`,
      y: `${(2 / 3) * 100}%`,
    });
  });
});
