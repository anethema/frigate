import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import LiveObjectOverlay from "./LiveObjectOverlay";

const overlay = {
  schema_version: 1 as const,
  camera: "front",
  frame_time: 100,
  detect_width: 1920,
  detect_height: 1080,
  autotracked_object_id: "person-1",
  objects: [
    {
      id: "person-1",
      frame_time: 100,
      label: "person",
      sub_label: "Alex",
      score: 0.876,
      stationary: false,
      box: [100, 200, 500, 800] as [number, number, number, number],
      autotracked: false,
    },
    {
      id: "car-1",
      frame_time: 100,
      label: "car",
      sub_label: null,
      score: 0.95,
      stationary: true,
      box: [600, 200, 1200, 800] as [number, number, number, number],
      autotracked: false,
    },
  ],
};

describe("LiveObjectOverlay", () => {
  it("renders detector-coordinate object and tracking styles", () => {
    const markup = renderToStaticMarkup(
      <LiveObjectOverlay
        overlay={overlay}
        expectedWidth={1920}
        expectedHeight={1080}
        colormap={{ person: [10, 20, 30] }}
      />,
    );

    expect(markup).toContain('viewBox="0 0 1920 1080"');
    expect(markup).toContain("TRACKING · person: Alex 88%");
    expect(markup).toContain('stroke-width="5"');
    expect(markup).toContain('stroke="rgb(30, 20, 10)"');
    expect(markup).toContain('vector-effect="non-scaling-stroke"');
    expect(markup).toContain('stroke="#9ca3af"');
    expect(markup).toContain('stroke-width="1"');
  });

  it("suppresses boxes when detector and player aspect ratios differ", () => {
    const markup = renderToStaticMarkup(
      <LiveObjectOverlay
        overlay={overlay}
        expectedWidth={1024}
        expectedHeight={1024}
      />,
    );

    expect(markup).toBe("");
  });
});
