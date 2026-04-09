"""
Generator-agnostic frequency heuristic detector.

Uses DCT/FFT spectral statistics trained via logistic regression.
Since the features are purely frequency-based (no semantic content),
this generalizes to unseen generators better than learned image classifiers.

The logistic regression is trained on the same data but uses ONLY
frequency features — no CLIP features. This makes it complementary
to the CLIP-based models.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.fft as fft
from PIL import Image
from scipy.ndimage import laplace


def extract_freq_features(image_pil: Image.Image) -> np.ndarray:
    """Extract frequency features from an image.

    Resizes to 224x224 first for consistency, then extracts
    a 13-dimensional feature vector. All operations are NaN-safe.

    Args:
        image_pil: Input PIL image (any size).

    Returns:
        Feature vector, shape (13,), guaranteed no NaN/Inf.
    """
    # Resize to standard size for consistent features
    image_pil = image_pil.convert("RGB").resize((224, 224), Image.LANCZOS)
    img = np.array(image_pil).astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    h, w = gray.shape

    try:
        # DCT features
        dct = fft.dctn(gray, type=2, norm="ortho")
        dct_abs = np.abs(dct)
        dct_log = np.log1p(dct_abs)

        q1, q2, q3 = h // 4, h // 2, 3 * h // 4
        low = float(dct_log[:q1, :q1].mean())
        mid_low = float(dct_log[q1:q2, q1:q2].mean())
        mid_high = float(dct_log[q2:q3, q2:q3].mean())
        high = float(dct_log[q3:, q3:].mean())
        rolloff = high / (low + 1e-8)
        mid_ratio = mid_high / (mid_low + 1e-8)

        # FFT radial profile + power law fit
        fft2d = np.fft.fft2(gray)
        fft_mag = np.abs(np.fft.fftshift(fft2d))
        fft_log = np.log1p(fft_mag)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = min(cy, cx)
        radial = np.zeros(max_r, dtype=np.float64)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial[i] = fft_log[mask].mean()

        # Power law fit with NaN protection
        valid = (radial[1:max_r] > 0) & np.isfinite(radial[1:max_r])
        if valid.sum() > 10:
            freqs = np.arange(1, max_r, dtype=np.float64)
            lf = np.log(freqs[valid])
            rad_valid = radial[1:max_r][valid]
            rad_valid = np.clip(rad_valid, 1e-15, None)
            lp = np.log(rad_valid)
            if np.all(np.isfinite(lf)) and np.all(np.isfinite(lp)):
                A = np.vstack([lf, np.ones(len(lf))]).T
                result = np.linalg.lstsq(A, lp, rcond=None)
                alpha, intercept = result[0]
                predicted = alpha * lf + intercept
                residual = float(np.sqrt(np.mean((lp - predicted) ** 2)))
                alpha = float(alpha)
            else:
                alpha, residual = -1.0, 0.0
        else:
            alpha, residual = -1.0, 0.0

        # Spectral entropy
        dct_flat = dct_abs.flatten()
        total = dct_flat.sum()
        if total > 0:
            dct_norm = dct_flat / total
            dct_norm = np.clip(dct_norm, 1e-15, None)
            entropy = float(-np.sum(dct_norm * np.log(dct_norm)))
        else:
            entropy = 0.0

        # Grid artifact score
        fft_ac = np.abs(np.fft.ifft2(fft_mag ** 2))
        dc = fft_ac[0, 0]
        if dc > 0:
            fft_ac = fft_ac / dc
        h8, w8 = max(h // 8, 2), max(w // 8, 2)
        center = fft_ac[1:h8, 1:w8]
        grid_score = float(center.max()) if center.size > 0 else 0.0

        # Noise stats
        noise_map = laplace(gray)
        noise_std = float(noise_map.std())
        std4 = noise_std ** 4
        if std4 > 1e-15:
            noise_kurtosis = float(
                np.mean((noise_map - noise_map.mean()) ** 4) / std4
            )
        else:
            noise_kurtosis = 3.0  # Normal distribution kurtosis

        # Channel entropy variance
        ch_ents = []
        for c in range(3):
            ch_dct = fft.dctn(img[:, :, c], type=2, norm="ortho")
            ch_flat = np.abs(ch_dct).flatten()
            ch_total = ch_flat.sum()
            if ch_total > 0:
                ch_n = ch_flat / ch_total
                ch_n = np.clip(ch_n, 1e-15, None)
                ch_ents.append(float(-np.sum(ch_n * np.log(ch_n))))
            else:
                ch_ents.append(0.0)
        ch_var = float(np.var(ch_ents))

    except Exception:
        # Fallback: return neutral features
        return np.zeros(13, dtype=np.float32)

    features = np.array(
        [
            low, mid_low, mid_high, high, rolloff, mid_ratio,
            alpha, residual, entropy, grid_score, ch_var,
            noise_std, noise_kurtosis,
        ],
        dtype=np.float32,
    )

    # Final NaN/Inf safety
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


class FreqHeuristicDetector:
    """Frequency-based detector using logistic regression.

    Train on frequency features extracted from real/fake images,
    then use for generator-agnostic detection.
    """

    def __init__(self) -> None:
        self.model = None
        self.scaler = None

    def train(self, data_dir: Path, n_samples: int = 4000) -> dict:
        """Train logistic regression on frequency features.

        Args:
            data_dir: Path to processed data directory.
            n_samples: Max samples to use (2K real + 2K fake).

        Returns:
            Training metrics dict.
        """
        import random
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from tqdm import tqdm

        random.seed(42)

        # Collect image paths
        real_paths = sorted((data_dir / "train" / "real").glob("*.jpg"))
        fake_paths = sorted((data_dir / "train" / "fake").glob("*.jpg"))

        per_class = n_samples // 2
        random.shuffle(real_paths)
        random.shuffle(fake_paths)
        real_paths = real_paths[:per_class]
        fake_paths = fake_paths[:per_class]

        print(f"Extracting freq features: {len(real_paths)} real + {len(fake_paths)} fake")

        # Extract features
        features = []
        labels = []
        for paths, label in [(real_paths, 0), (fake_paths, 1)]:
            for p in tqdm(paths, desc=f"{'Real' if label == 0 else 'Fake'}"):
                img = Image.open(p).convert("RGB")
                feat = extract_freq_features(img)
                features.append(feat)
                labels.append(label)

        X = np.array(features)
        y = np.array(labels)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression
        self.model = LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        )
        self.model.fit(X_scaled, y)

        # Training metrics
        train_probs = self.model.predict_proba(X_scaled)[:, 1]
        train_acc = self.model.score(X_scaled, y)
        train_auc = roc_auc_score(y, train_probs)

        print(f"Freq heuristic trained: acc={train_acc:.4f}, auc={train_auc:.4f}")
        return {"accuracy": train_acc, "auc": train_auc}

    def save(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: Path) -> bool:
        """Load model from disk. Returns True if successful."""
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
        return True

    def predict(self, image_pil: Image.Image) -> tuple[float, list[str]]:
        """Predict AI probability using frequency features.

        Args:
            image_pil: Input PIL image.

        Returns:
            Tuple of (ai_probability, list of reason strings).
        """
        if self.model is None:
            return 0.5, ["Frequency detector not trained"]

        feat = extract_freq_features(image_pil).reshape(1, -1)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat_scaled = self.scaler.transform(feat)
        feat_scaled = np.nan_to_num(feat_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        prob = self.model.predict_proba(feat_scaled)[0, 1]

        # Generate reasons based on feature importance
        reasons = []
        coefs = self.model.coef_[0]
        feat_names = [
            "low_freq", "mid_low_freq", "mid_high_freq", "high_freq",
            "freq_rolloff", "mid_ratio", "power_law_alpha", "power_law_residual",
            "spectral_entropy", "grid_score", "channel_entropy_var",
            "noise_std", "noise_kurtosis",
        ]

        # Top contributing features
        contributions = coefs * feat_scaled[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]
        for idx in top_idx:
            direction = "elevated" if contributions[idx] > 0 else "reduced"
            reasons.append(f"{feat_names[idx].replace('_', ' ')} is {direction}")

        return float(prob), reasons


# Module-level convenience function
_detector = None


def freq_heuristic_score(image_pil: Image.Image) -> tuple[float, list[str]]:
    """Compute frequency-based AI detection score.

    Loads the trained model on first call. If no trained model exists,
    returns 0.5 (uncertain).

    Args:
        image_pil: Input PIL image.

    Returns:
        Tuple of (ai_probability, reasons).
    """
    global _detector
    if _detector is None:
        _detector = FreqHeuristicDetector()
        model_path = Path(__file__).parent.parent / "checkpoints" / "freq_heuristic.pkl"
        if not _detector.load(model_path):
            return 0.5, ["Frequency detector not available"]

    return _detector.predict(image_pil)
