import math
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from ..base import FrigateBaseModel
from ..env import EnvString
from .objects import DEFAULT_TRACKED_OBJECTS

__all__ = ["OnvifConfig", "PtzAutotrackConfig", "ZoomingModeEnum"]


class ZoomingModeEnum(str, Enum):
    disabled = "disabled"
    absolute = "absolute"
    relative = "relative"


class PtzAutotrackConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enable PTZ object autotracking.")
    movement_status: Literal["onvif", "position"] = Field(
        default="onvif",
        title="Movement status source. Position is an experimental PTZ fallback.",
    )
    preset_movement: Literal["preset", "absolute"] = Field(
        default="preset",
        title="Use saved preset coordinates when native preset recall is broken.",
    )
    return_preset_position: Optional[list[float]] = Field(
        default=None,
        min_length=3,
        max_length=3,
        title="Verified native return-preset pan, tilt, zoom when preset metadata is incorrect.",
    )
    calibrate_on_startup: bool = Field(
        default=False, title="Perform a camera calibration when Frigate starts."
    )
    zooming: ZoomingModeEnum = Field(
        default=ZoomingModeEnum.disabled, title="Autotracker zooming mode."
    )
    position_zoom_limits: Optional[list[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        title="Verified reachable zoom limits in normalized ONVIF coordinates for position status.",
    )
    zoom_factor: float = Field(
        default=0.3,
        title="Zooming factor (0.1-0.75).",
        ge=0.1,
        le=0.75,
    )
    track: list[str] = Field(default=DEFAULT_TRACKED_OBJECTS, title="Objects to track.")
    required_zones: list[str] = Field(
        default_factory=list,
        title="List of required zones to be entered in order to begin autotracking.",
    )
    return_preset: str = Field(
        default="home",
        title="Name of camera preset to return to when object tracking is over.",
    )
    timeout: int = Field(
        default=10, title="Seconds to delay before returning to preset."
    )
    movement_weights: Optional[Union[str, list[str]]] = Field(
        default_factory=list,
        title="Internal value used for PTZ movements based on the speed of your camera's motor.",
    )
    enabled_in_config: Optional[bool] = Field(
        default=None, title="Keep track of original state of autotracking."
    )

    @model_validator(mode="after")
    def validate_position_status(self):
        if self.movement_status == "position" and self.zooming == ZoomingModeEnum.relative:
            raise ValueError("Position movement status supports zooming: disabled or absolute")
        if self.position_zoom_limits is not None:
            low, high = self.position_zoom_limits
            if self.movement_status != "position" or not all(math.isfinite(x) for x in (low, high)) or not 0 <= low < high <= 1:
                raise ValueError("Position zoom limits require position status and 0 <= min < max <= 1")
        if self.preset_movement == "absolute" and self.movement_status != "position":
            raise ValueError("Absolute preset recall requires position movement status")
        if self.return_preset_position is not None:
            pan, tilt, zoom = self.return_preset_position
            if self.movement_status != "position" or not all(math.isfinite(x) for x in (pan, tilt, zoom)):
                raise ValueError("Verified preset position requires position status and finite coordinates")
            if not (-1 <= pan <= 1 and -1 <= tilt <= 1 and 0 <= zoom <= 1):
                raise ValueError("Verified preset position must use normalized generic coordinates")
        return self

    @field_validator("movement_weights", mode="before")
    @classmethod
    def validate_weights(cls, v):
        if v is None:
            return None

        if isinstance(v, str):
            weights = list(map(str, map(float, v.split(","))))
        elif isinstance(v, list):
            weights = [str(float(val)) for val in v]
        else:
            raise ValueError("Invalid type for movement_weights")

        if len(weights) != 6:
            raise ValueError(
                "movement_weights must have exactly 6 floats, remove this line from your config and run autotracking calibration"
            )

        return weights


class OnvifConfig(FrigateBaseModel):
    host: EnvString = Field(default="", title="Onvif Host")
    port: int = Field(default=8000, title="Onvif Port")
    user: Optional[EnvString] = Field(default=None, title="Onvif Username")
    password: Optional[EnvString] = Field(default=None, title="Onvif Password")
    tls_insecure: bool = Field(default=False, title="Onvif Disable TLS verification")
    autotracking: PtzAutotrackConfig = Field(
        default_factory=PtzAutotrackConfig,
        title="PTZ auto tracking config.",
    )
    ignore_time_mismatch: bool = Field(
        default=False,
        title="Onvif Ignore Time Synchronization Mismatch Between Camera and Server",
    )
