import {
  LiveObject,
  LiveObjectOverlay as LiveObjectOverlayData,
} from "@/types/live";

type LiveObjectOverlayProps = {
  overlay?: LiveObjectOverlayData;
  expectedWidth: number;
  expectedHeight: number;
  colormap?: Record<string, [number, number, number]>;
};

const ASPECT_RATIO_TOLERANCE = 0.01;

function getBox(object: LiveObject, width: number, height: number) {
  const [rawLeft, rawTop, rawRight, rawBottom] = object.box;
  const left = Math.max(0, Math.min(rawLeft, width));
  const top = Math.max(0, Math.min(rawTop, height));
  const right = Math.max(0, Math.min(rawRight, width));
  const bottom = Math.max(0, Math.min(rawBottom, height));

  if (right <= left || bottom <= top) {
    return undefined;
  }

  return { left, top, width: right - left, height: bottom - top };
}

function formatObjectLabel(object: LiveObject, tracking: boolean) {
  const confidence = `${Math.round(object.score * 100)}%`;
  const label = object.sub_label
    ? `${object.label}: ${object.sub_label}`
    : object.label;
  return `${tracking ? "TRACKING · " : ""}${label} ${confidence}`;
}

function getObjectColor(
  label: string,
  colormap?: Record<string, [number, number, number]>,
) {
  const bgrColor = colormap?.[label];
  return bgrColor
    ? `rgb(${bgrColor[2]}, ${bgrColor[1]}, ${bgrColor[0]})`
    : "hsl(var(--selected))";
}

/** Renders detector-coordinate boxes without intercepting video controls. */
export default function LiveObjectOverlay({
  overlay,
  expectedWidth,
  expectedHeight,
  colormap,
}: LiveObjectOverlayProps) {
  if (
    !overlay ||
    overlay.objects.length === 0 ||
    overlay.detect_width <= 0 ||
    overlay.detect_height <= 0 ||
    expectedWidth <= 0 ||
    expectedHeight <= 0
  ) {
    return null;
  }

  const detectorAspectRatio = overlay.detect_width / overlay.detect_height;
  const expectedAspectRatio = expectedWidth / expectedHeight;

  // A different detector frame means these coordinates would be mapped onto
  // the wrong pixels. Hide the overlay rather than drawing misleading boxes.
  if (
    Math.abs(detectorAspectRatio - expectedAspectRatio) > ASPECT_RATIO_TOLERANCE
  ) {
    return null;
  }

  return (
    <svg
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-30 size-full"
      viewBox={`0 0 ${overlay.detect_width} ${overlay.detect_height}`}
      preserveAspectRatio="none"
    >
      {overlay.objects.map((object) => {
        const box = getBox(object, overlay.detect_width, overlay.detect_height);
        if (!box) {
          return null;
        }

        const tracking =
          object.autotracked || overlay.autotracked_object_id === object.id;
        const color =
          tracking || !object.stationary
            ? getObjectColor(object.label, colormap)
            : "#9ca3af";
        const strokeWidth = tracking ? 5 : object.stationary ? 1 : 2;
        const label = formatObjectLabel(object, tracking);
        const labelY = Math.max(14, box.top - 6);

        return (
          <g key={object.id}>
            <rect
              x={box.left}
              y={box.top}
              width={box.width}
              height={box.height}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={box.left}
              y={labelY}
              fill="white"
              fontSize="14"
              fontWeight="600"
              paintOrder="stroke"
              stroke="rgba(0, 0, 0, 0.8)"
              strokeWidth="3"
              strokeLinejoin="round"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
