"""Live tracked-object data for native camera overlays."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from frigate.api.auth import require_camera_access
from frigate.api.defs.tags import Tags

router = APIRouter(tags=[Tags.camera])


def _normalize_sub_label(sub_label: Any) -> str | None:
    """Expose a label string without the internal sub-label confidence tuple."""
    if isinstance(sub_label, str):
        return sub_label
    if isinstance(sub_label, (list, tuple)) and sub_label:
        return sub_label[0] if isinstance(sub_label[0], str) else None
    return None


def _get_autotracked_object_id(
    camera_name: str, detected_frames_processor
) -> str | None:
    """Return the current autotracked object's id when PTZ tracking is available."""
    try:
        autotracker = detected_frames_processor.ptz_autotracker_thread.ptz_autotracker
        tracked_object = autotracker.tracked_object.get(camera_name)
        if tracked_object is None:
            return None
        return tracked_object.obj_data.get("id")
    except (AttributeError, KeyError, TypeError):
        return None


def _serialize_object(object_data: dict[str, Any], stationary: bool, autotracked: bool):
    """Serialize only the fields required to draw a current-frame overlay."""
    return {
        "id": object_data["id"],
        "frame_time": object_data["frame_time"],
        "label": object_data["label"],
        "sub_label": _normalize_sub_label(object_data.get("sub_label")),
        "score": object_data["score"],
        "stationary": stationary,
        "box": list(object_data["box"]),
        "autotracked": autotracked,
    }


@router.get(
    "/live/{camera_name}/objects",
    dependencies=[Depends(require_camera_access)],
)
def get_live_objects(request: Request, camera_name: str):
    """Return current tracked objects in the camera's detect coordinate space."""
    camera_config = request.app.frigate_config.cameras.get(camera_name)
    if camera_config is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")

    detected_frames_processor = request.app.detected_frames_processor
    camera_state = getattr(detected_frames_processor, "camera_states", {}).get(
        camera_name
    )
    frame_time = 0.0
    tracked_objects: list[tuple[dict[str, Any], bool]] = []

    if camera_state is not None:
        # Copy just the values needed for the response while the current frame and
        # tracked-object mapping are consistent. JSON serialization happens after
        # releasing this lock so frame processing is not blocked by the response.
        with camera_state.current_frame_lock:
            frame_time = camera_state.current_frame_time
            tracked_objects = [
                (
                    {
                        "id": tracked_object.obj_data["id"],
                        "frame_time": tracked_object.obj_data["frame_time"],
                        "label": tracked_object.obj_data["label"],
                        "sub_label": tracked_object.obj_data.get("sub_label"),
                        "score": tracked_object.obj_data["score"],
                        "box": list(tracked_object.obj_data["box"]),
                    },
                    not tracked_object.active,
                )
                for tracked_object in camera_state.tracked_objects.values()
                if not tracked_object.false_positive
                and tracked_object.obj_data.get("frame_time") == frame_time
            ]

    autotracked_object_id = _get_autotracked_object_id(
        camera_name, detected_frames_processor
    )
    objects = [
        _serialize_object(
            object_data,
            stationary,
            object_data["id"] == autotracked_object_id,
        )
        for object_data, stationary in tracked_objects
    ]

    return JSONResponse(
        content={
            "schema_version": 1,
            "camera": camera_name,
            "detect_width": camera_config.detect.width,
            "detect_height": camera_config.detect.height,
            "frame_time": frame_time,
            "autotracked_object_id": autotracked_object_id,
            "objects": objects,
        },
        headers={"Cache-Control": "no-store"},
    )
