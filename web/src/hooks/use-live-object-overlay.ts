import axios from "axios";
import { useEffect, useRef, useState } from "react";
import { LiveObjectOverlay, LivePlayerMode } from "@/types/live";

const POLL_INTERVAL_MS = 200;
const STALE_TIMEOUT_MS = 1000;

type UseLiveObjectOverlayOptions = {
  camera: string;
  enabled: boolean;
  liveReady: boolean;
  liveMode: LivePlayerMode;
};

function isLiveObjectOverlay(value: unknown): value is LiveObjectOverlay {
  if (!value || typeof value !== "object") {
    return false;
  }

  const response = value as Partial<LiveObjectOverlay>;
  return (
    response.schema_version === 1 &&
    typeof response.camera === "string" &&
    typeof response.frame_time === "number" &&
    typeof response.detect_width === "number" &&
    typeof response.detect_height === "number" &&
    Array.isArray(response.objects)
  );
}

function isBrowserActive() {
  return document.visibilityState === "visible" && navigator.onLine;
}

/**
 * Poll the lightweight live-object endpoint only while a native video player is
 * actually presenting a visible, online stream. A request is never started
 * until the previous one has completed.
 */
export function useLiveObjectOverlay({
  camera,
  enabled,
  liveReady,
  liveMode,
}: UseLiveObjectOverlayOptions): LiveObjectOverlay | undefined {
  const [browserActive, setBrowserActive] = useState(isBrowserActive);
  const [overlay, setOverlay] = useState<LiveObjectOverlay>();
  const lastFrameTimeRef = useRef<number>();
  const abortControllerRef = useRef<AbortController>();

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const updateBrowserActive = () => setBrowserActive(isBrowserActive());
    updateBrowserActive();

    document.addEventListener("visibilitychange", updateBrowserActive);
    window.addEventListener("online", updateBrowserActive);
    window.addEventListener("offline", updateBrowserActive);

    return () => {
      document.removeEventListener("visibilitychange", updateBrowserActive);
      window.removeEventListener("online", updateBrowserActive);
      window.removeEventListener("offline", updateBrowserActive);
    };
  }, [enabled]);

  const active =
    enabled &&
    liveReady &&
    browserActive &&
    (liveMode === "mse" || liveMode === "webrtc");

  useEffect(() => {
    // Discard the previous camera or playback state before a new polling loop
    // can publish results.
    abortControllerRef.current?.abort();
    abortControllerRef.current = undefined;
    lastFrameTimeRef.current = undefined;
    setOverlay(undefined);
  }, [active, camera]);

  useEffect(() => {
    if (!active) {
      return;
    }

    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    let staleTimer: ReturnType<typeof setTimeout> | undefined;

    const clearOverlay = () => {
      staleTimer = undefined;
      if (!cancelled) {
        setOverlay(undefined);
      }
    };

    const scheduleStaleClear = (restart = false) => {
      if (staleTimer && !restart) {
        return;
      }
      if (staleTimer) {
        clearTimeout(staleTimer);
      }
      staleTimer = setTimeout(clearOverlay, STALE_TIMEOUT_MS);
    };

    const poll = async () => {
      if (cancelled) {
        return;
      }

      const pollStartedAt = Date.now();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const response = await axios.get<unknown>(`live/${camera}/objects`, {
          signal: controller.signal,
          timeout: STALE_TIMEOUT_MS,
        });

        if (cancelled || controller.signal.aborted) {
          return;
        }

        if (
          !isLiveObjectOverlay(response.data) ||
          response.data.camera !== camera
        ) {
          scheduleStaleClear();
          return;
        }

        if (
          lastFrameTimeRef.current !== undefined &&
          response.data.frame_time <= lastFrameTimeRef.current
        ) {
          return;
        }

        lastFrameTimeRef.current = response.data.frame_time;
        // Empty frames must remove boxes immediately instead of retaining the
        // previous frame until the stale timeout.
        setOverlay(response.data);
        scheduleStaleClear(true);
      } catch (error) {
        if (!axios.isCancel(error) && !cancelled) {
          // Keep a successful frame briefly during a transient failure, but
          // never let a stale overlay survive for more than one second.
          scheduleStaleClear();
        }
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = undefined;
        }

        if (!cancelled) {
          pollTimer = setTimeout(
            poll,
            Math.max(0, POLL_INTERVAL_MS - (Date.now() - pollStartedAt)),
          );
        }
      }
    };

    poll();

    return () => {
      cancelled = true;
      abortControllerRef.current?.abort();
      abortControllerRef.current = undefined;
      if (pollTimer) {
        clearTimeout(pollTimer);
      }
      if (staleTimer) {
        clearTimeout(staleTimer);
      }
    };
  }, [active, camera]);

  return overlay;
}
