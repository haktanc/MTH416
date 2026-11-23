"""Bound the temporary memory of dense fold-local standardization."""

from sklearn.preprocessing import StandardScaler


class BatchedStandardScaler(StandardScaler):
    """Use StandardScaler's incremental statistics over a single fit fold."""

    def fit(self, X, y=None, sample_weight=None):
        if len(X) == 0:
            return super().fit(X, y, sample_weight=sample_weight)
        self._reset()
        for start in range(0, len(X), 16):
            batch = slice(start, start + 16)
            weights = None if sample_weight is None else sample_weight[batch]
            self.partial_fit(X[batch], sample_weight=weights)
        return self


def weight_glcm(X, factor):
    """Apply a fixed multiplier to the last 48 columns after standardization."""
    X[:, -48:] *= factor
    return X
