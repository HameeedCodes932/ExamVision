from app.services.gaze.models import GazeTarget, GazeVector
from app.services.gaze.utils import (
    classify_gaze_target,
    compute_gaze_from_iris,
)


class TestGazeTarget:
    def test_values(self) -> None:
        assert GazeTarget.LOOKING_FORWARD.value == "looking_forward"
        assert GazeTarget.LOOKING_LEFT.value == "looking_left"
        assert GazeTarget.LOOKING_RIGHT.value == "looking_right"
        assert GazeTarget.LOOKING_AT_PAPER.value == "looking_at_paper"
        assert GazeTarget.LOOKING_BEHIND.value == "looking_behind"
        assert GazeTarget.LOOKING_AT_ANOTHER_STUDENT.value == "looking_at_another_student"


class TestGazeVector:
    def test_forward_default(self) -> None:
        gv = GazeVector(gaze_yaw=0.0, gaze_pitch=0.0)
        assert gv.target == GazeTarget.LOOKING_FORWARD
        assert gv.target_label == "looking_forward"

    def test_to_dict(self) -> None:
        gv = GazeVector(gaze_yaw=15.123, gaze_pitch=-5.456, target=GazeTarget.LOOKING_LEFT)
        d = gv.to_dict()
        assert d["gaze_yaw"] == 15.12
        assert d["gaze_pitch"] == -5.46
        assert d["target"] == "looking_left"


class TestClassifyGazeTarget:
    def test_forward(self) -> None:
        assert classify_gaze_target(0, 0) == GazeTarget.LOOKING_FORWARD

    def test_left(self) -> None:
        assert classify_gaze_target(30, 0) == GazeTarget.LOOKING_LEFT

    def test_right(self) -> None:
        assert classify_gaze_target(-30, 0) == GazeTarget.LOOKING_RIGHT

    def test_behind(self) -> None:
        assert classify_gaze_target(60, 0) == GazeTarget.LOOKING_BEHIND

    def test_behind_negative(self) -> None:
        assert classify_gaze_target(-60, 0) == GazeTarget.LOOKING_BEHIND

    def test_at_paper(self) -> None:
        assert classify_gaze_target(0, 30) == GazeTarget.LOOKING_AT_PAPER

    def test_at_paper_with_yaw(self) -> None:
        assert classify_gaze_target(10, 30) == GazeTarget.LOOKING_AT_PAPER

    def test_forward_edge_yaw(self) -> None:
        assert classify_gaze_target(19, 0) == GazeTarget.LOOKING_FORWARD

    def test_forward_edge_pitch(self) -> None:
        assert classify_gaze_target(0, 14) == GazeTarget.LOOKING_FORWARD

    def test_left_not_behind(self) -> None:
        assert classify_gaze_target(40, 0) == GazeTarget.LOOKING_LEFT

    def test_right_not_behind(self) -> None:
        assert classify_gaze_target(-40, 0) == GazeTarget.LOOKING_RIGHT


class TestComputeGazeFromIris:
    def test_center_gaze(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (50, 0), head_yaw=0, head_pitch=0)
        assert abs(yaw) < 1
        assert abs(pitch) < 1

    def test_left_gaze(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (25, 0), head_yaw=0, head_pitch=0)
        assert yaw < 0
        assert abs(pitch) < 1

    def test_right_gaze(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (75, 0), head_yaw=0, head_pitch=0)
        assert yaw > 0
        assert abs(pitch) < 1

    def test_up_gaze(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (50, -10), head_yaw=0, head_pitch=0)
        assert pitch > 0

    def test_down_gaze(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (50, 10), head_yaw=0, head_pitch=0)
        assert pitch < 0

    def test_with_head_pose(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 100, 0), (50, 0), head_yaw=30, head_pitch=10)
        assert abs(yaw - 30) < 1
        assert abs(pitch - 10) < 1

    def test_zero_width(self) -> None:
        yaw, pitch = compute_gaze_from_iris((0, 0, 0, 0), (0, 0), head_yaw=15, head_pitch=5)
        assert yaw == 15
        assert pitch == 5
