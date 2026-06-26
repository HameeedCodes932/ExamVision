import numpy as np

NDIM = 4
STATE_DIM = 8
MEASURE_DIM = 4


def _get_std(dt: float = 1.0) -> tuple[float, float, float, float]:
    std_pos = 0.05 * dt
    std_vel = 0.00625 * dt
    return (std_pos, std_pos, std_pos * 2, std_vel * 10)


class KalmanFilter:
    def __init__(self) -> None:
        self._motion_mat: np.ndarray = np.eye(STATE_DIM, dtype=float)
        for i in range(NDIM):
            self._motion_mat[i, NDIM + i] = 1.0

        self._update_mat: np.ndarray = np.eye(MEASURE_DIM, STATE_DIM, dtype=float)

        self._std_weight_position = 0.05
        self._std_weight_velocity = 0.00625

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(STATE_DIM, dtype=float)
        mean[:NDIM] = measurement
        mean[NDIM:] = 0.0

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[2],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = 1.0
        motion = self._motion_mat.copy()
        motion[:NDIM, NDIM:] *= dt

        std_pos = self._std_weight_position * dt
        std_vel = self._std_weight_velocity * dt
        noise = np.diag(
            np.square(
                [
                    std_pos * mean[2],
                    std_pos * mean[3],
                    1e-2,
                    std_pos * mean[2],
                    std_vel * mean[2],
                    std_vel * mean[3],
                    1e-5,
                    std_vel * mean[2],
                ]
            )
        )

        mean_pred = motion @ mean
        cov_pred = motion @ covariance @ motion.T + noise
        return mean_pred, cov_pred

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[2],
        ]
        noise = np.diag(np.square(std))
        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ covariance @ self._update_mat.T + noise
        return mean_proj, cov_proj

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_proj, cov_proj = self.project(mean, covariance)
        innovation = measurement - mean_proj
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(cov_proj)
        mean_new = mean + kalman_gain @ innovation
        cov_new = covariance - kalman_gain @ self._update_mat @ covariance
        return mean_new, cov_new

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = measurements[:, :2]

        cholesky = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = np.linalg.solve(cholesky, d.T)
        return np.sum(z * z, axis=0)
