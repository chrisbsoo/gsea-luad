from dataclasses import dataclass, field
from typing import Literal
import pandas as pd

# NOTE: result wrapper as an object, no awkward unpacking needed

@dataclass
class resultROMA:

    scores: pd.DataFrame
    l1: pd.Series
    l1_pval: pd.Series
    coord_pval: pd.Series
    gene_weights: dict[str, pd.Series] = field(default_factory=dict)
    dropped_samples: dict[str, list[str]] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05, by: Literal["l1", "coord", "both"] = "l1") -> pd.DataFrame:
        """
        INPUT: alpha (sig level), by (which p-value columns to filter on)
        OUTPUT: all three columns (L1, l1_pval, coord_pval), but only for the
        modules that pass the alpha threshold on the chosen columns.
        """
        valid = {"l1", "coord", "both"}
        if by not in valid:
            raise ValueError(f"by must be one of {valid}, got {by!r} instead.")

        result = pd.DataFrame({"l1": self.l1, "l1_pval": self.l1_pval, "coord": self.coord_pval})

        if by == "l1":
            mask = result["l1_pval"] < alpha
            return result[mask].sort_values("l1_pval")
        elif by == "coord":
            mask = result["coord"] < alpha
            return result[mask].sort_values("coord")
        else:
            mask = (result["l1_pval"] < alpha) & (result["coord"] < alpha)
            return result[mask].sort_values("l1_pval")

    def __repr__(self) -> str:
        n_modules, n_samples = self.scores.shape
        n_sig = int((self.l1_pval < 0.05).sum())
        return f"resultROMA({n_modules} modules x {n_samples} samples, {n_sig} significant at l1_pval<0.05)"