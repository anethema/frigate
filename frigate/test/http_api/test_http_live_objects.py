"""Tests for the current tracked-object overlay endpoint."""

import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from frigate.test.http_api.base_http_test import AuthTestClient, BaseTestHttp


def _tracked_object(
    object_id: str,
    frame_time: float,
    *,
    false_positive: bool = False,
    active: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        obj_data={
            "id": object_id,
            "frame_time": frame_time,
            "label": "person",
            "sub_label": ["visitor", 0.91],
            "score": 0.87,
            "box": [12, 34, 56, 78],
        },
        false_positive=false_positive,
        active=active,
    )


class TestHttpLiveObjects(BaseTestHttp):
    def setUp(self):
        super().setUp([])
        self.app = super().create_app()

        current = _tracked_object("current", 1234.5, active=False)
        stale = _tracked_object("stale", 1234.0)
        false_positive = _tracked_object("false-positive", 1234.5, false_positive=True)
        camera_state = SimpleNamespace(
            current_frame_lock=threading.Lock(),
            current_frame_time=1234.5,
            tracked_objects={
                "current": current,
                "stale": stale,
                "false-positive": false_positive,
            },
        )
        autotracker = SimpleNamespace(tracked_object={"front_door": current})
        self.app.detected_frames_processor = SimpleNamespace(
            camera_states={"front_door": camera_state},
            ptz_autotracker_thread=SimpleNamespace(ptz_autotracker=autotracker),
        )

    def test_returns_only_current_non_false_positive_objects(self):
        with AuthTestClient(self.app) as client:
            response = client.get("/live/front_door/objects")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "schema_version": 1,
            "camera": "front_door",
            "detect_width": 1920,
            "detect_height": 1080,
            "frame_time": 1234.5,
            "autotracked_object_id": "current",
            "objects": [
                {
                    "id": "current",
                    "frame_time": 1234.5,
                    "label": "person",
                    "sub_label": "visitor",
                    "score": 0.87,
                    "stationary": True,
                    "box": [12, 34, 56, 78],
                    "autotracked": True,
                }
            ],
        }

    def test_returns_empty_snapshot_without_camera_state(self):
        self.app.detected_frames_processor.camera_states = {}
        self.app.detected_frames_processor.ptz_autotracker_thread = None

        with AuthTestClient(self.app) as client:
            response = client.get("/live/front_door/objects")

        assert response.status_code == 200
        assert response.json()["frame_time"] == 0.0
        assert response.json()["objects"] == []

    def test_requires_authentication_and_camera_access(self):
        with TestClient(self.app) as client:
            unauthenticated = client.get("/live/front_door/objects")

        assert unauthenticated.status_code == 401

        self.app.frigate_config.auth.roles["front_door_only"] = ["front_door"]
        self.app.frigate_config.auth.roles["no_access"] = ["another_camera"]
        with AuthTestClient(self.app) as client:
            allowed = client.get(
                "/live/front_door/objects",
                headers={
                    "remote-user": "allowed-user",
                    "remote-role": "front_door_only",
                },
            )
            denied = client.get(
                "/live/front_door/objects",
                headers={"remote-user": "denied-user", "remote-role": "no_access"},
            )

        assert allowed.status_code == 200
        assert denied.status_code == 403

    def test_returns_objects_when_autotracker_is_unavailable(self):
        self.app.detected_frames_processor.ptz_autotracker_thread = None

        with AuthTestClient(self.app) as client:
            response = client.get("/live/front_door/objects")

        assert response.status_code == 200
        assert response.json()["objects"][0]["autotracked"] is False

    def test_normalizes_string_and_missing_sub_labels(self):
        current = self.app.detected_frames_processor.camera_states[
            "front_door"
        ].tracked_objects["current"]
        current.obj_data["sub_label"] = "known_person"

        with AuthTestClient(self.app) as client:
            string_response = client.get("/live/front_door/objects")

        assert string_response.json()["objects"][0]["sub_label"] == "known_person"

        del current.obj_data["sub_label"]
        with AuthTestClient(self.app) as client:
            missing_response = client.get("/live/front_door/objects")

        assert missing_response.json()["objects"][0]["sub_label"] is None

    def test_unknown_camera_returns_not_found(self):
        with AuthTestClient(self.app) as client:
            response = client.get("/live/unknown/objects")

        assert response.status_code == 404
