"""
Group-disjoint train/test (and optional val) splitting for TS-GNN.

Context
-------
Each cell line or patient gets ONE TP53 mutation label that is propagated to
ALL of its cells.  If cells from the same cell line appear in both train and
test, the model can memorise cell-line identity rather than TP53 biology
(group leakage).  Ravasio 2023 BSc thesis hit 100 % test accuracy for exactly
this reason.  The splitter and assertion in this module make group-disjoint
splits the only path forward.

References
----------
* Project spec tp53_gnn_project.md §3.3:
  "Split by cell line or patient to avoid information leakage."
* scikit-learn GroupShuffleSplit / StratifiedGroupKFold
"""

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class LeakageError(Exception):
    """Raised when the same group id appears in more than one split."""


# ─────────────────────────────────────────────────────────────────────────────
# Assertion helper
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_group_leakage(
    train_groups: Sequence,
    test_groups: Sequence,
    val_groups: Sequence | None = None,
) -> None:
    """Assert that no group id is shared between splits.

    Parameters
    ----------
    train_groups:
        Group ids of training samples (e.g. cell-line ids).
    test_groups:
        Group ids of test samples.
    val_groups:
        Optional group ids of validation samples.

    Raises
    ------
    LeakageError
        If any group id appears in more than one split.  The message
        lists the offending ids and the pairs of splits where they appear.

    Notes
    -----
    Empty inputs (e.g. empty lists) pass silently by design — an empty
    split cannot contain overlapping groups.
    """
    sets: dict[str, set] = {
        "train": set(train_groups),
        "test": set(test_groups),
    }
    if val_groups is not None:
        sets["val"] = set(val_groups)

    split_names = list(sets.keys())
    violations: list[str] = []

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = sets[a] & sets[b]
            if overlap:
                violations.append(
                    f"Groups {sorted(overlap)} appear in both '{a}' and '{b}'"
                )

    if violations:
        detail = "; ".join(violations)
        raise LeakageError(
            f"Group leakage detected — the following group ids are shared across "
            f"splits: {detail}.  Split by cell line / patient before proceeding."
        )

    logger.debug(
        "assert_no_group_leakage: OK — %d train / %d test%s unique group ids",
        len(sets["train"]),
        len(sets["test"]),
        f" / {len(sets['val'])} val" if val_groups is not None else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Splitter
# ─────────────────────────────────────────────────────────────────────────────

def group_split(
    groups: Sequence,
    labels: Sequence,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """Return group-disjoint sample indices for train / (val /) test splits.

    No group id will appear in more than one split.  The split is
    deterministic given ``random_seed``.

    Parameters
    ----------
    groups:
        Per-sample group ids (e.g. cell-line ids or patient ids).  Must be
        the same length as ``labels``.
    labels:
        Per-sample class labels.  Used for logging only (stratification at
        the group level is infeasible with GroupShuffleSplit when some groups
        have fewer than 2 samples).
    test_size:
        Fraction of *groups* to assign to the test set.  Must be in (0, 1).
    val_size:
        If provided, an additional validation set is carved out of the
        training remainder.  The fraction is relative to the *full* dataset,
        so ``train_size ≈ 1 − test_size − val_size``.  Must be in (0, 1)
        and ``test_size + val_size`` must be < 1.
    random_seed:
        Integer seed for reproducibility.  The test split uses ``random_seed``
        and the val split (when requested) uses ``random_seed + 1`` to avoid
        correlation between the two splits.

    Returns
    -------
    Two-way split (val_size is None):
        ``(train_idx, test_idx)`` — arrays of integer sample indices.
    Three-way split (val_size is not None):
        ``(train_idx, val_idx, test_idx)``.

    Raises
    ------
    LeakageError
        Via :func:`assert_no_group_leakage` if the produced split is somehow
        not group-disjoint (should never happen, but checked as a guardrail).
    ValueError
        If ``groups`` has fewer than 2 unique ids, if ``test_size`` is not in
        (0, 1), if ``val_size`` is given and not in (0, 1), if
        ``test_size + val_size >= 1.0``, or if ``groups`` / ``labels``
        lengths mismatch.
    """
    groups = np.asarray(groups)
    labels = np.asarray(labels)

    if len(groups) != len(labels):
        raise ValueError(
            f"groups and labels must have the same length "
            f"({len(groups)} vs {len(labels)})"
        )

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 2:
        raise ValueError(
            f"groups must contain at least 2 unique group ids; "
            f"got {n_unique_groups}."
        )

    if not (0 < test_size < 1):
        raise ValueError(
            f"test_size must be in the open interval (0, 1); got {test_size!r}."
        )

    if val_size is not None:
        if not (0 < val_size < 1):
            raise ValueError(
                f"val_size must be in the open interval (0, 1); got {val_size!r}."
            )
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0"
            )

    n_samples = len(groups)
    all_idx = np.arange(n_samples)

    # ── Step 1: carve out test set ──────────────────────────────────────────
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    trainval_idx, test_idx = next(gss_test.split(all_idx, labels, groups=groups))

    if val_size is None:
        train_idx = trainval_idx
        # Integrity check
        assert_no_group_leakage(
            groups[train_idx], groups[test_idx]
        )
        logger.info(
            "group_split: train=%d samples (%d groups), test=%d samples (%d groups)",
            len(train_idx),
            len(np.unique(groups[train_idx])),
            len(test_idx),
            len(np.unique(groups[test_idx])),
        )
        return train_idx, test_idx

    # ── Step 2: carve out val set from trainval ─────────────────────────────
    # val_size is relative to full dataset; adjust to fraction of trainval
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction_of_trainval,
        random_state=random_seed + 1,  # different seed to avoid correlation
    )
    trainval_groups = groups[trainval_idx]
    trainval_labels = labels[trainval_idx]
    local_train, local_val = next(
        gss_val.split(
            np.arange(len(trainval_idx)),
            trainval_labels,
            groups=trainval_groups,
        )
    )
    train_idx = trainval_idx[local_train]
    val_idx = trainval_idx[local_val]

    # Integrity check
    assert_no_group_leakage(groups[train_idx], groups[test_idx], groups[val_idx])

    logger.info(
        "group_split: train=%d samples (%d groups), val=%d samples (%d groups), "
        "test=%d samples (%d groups)",
        len(train_idx), len(np.unique(groups[train_idx])),
        len(val_idx), len(np.unique(groups[val_idx])),
        len(test_idx), len(np.unique(groups[test_idx])),
    )
    return train_idx, val_idx, test_idx
