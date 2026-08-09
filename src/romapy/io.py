from __future__ import annotations
from pathlib import Path

# NOTE: gmt file transformer to key-value data structure for pre-existing pathways

def load_gmt(path: str | Path) -> dict[str, list[str]]:

    """
        INPUT: address, i.e "data/hallmark.gmt"
        OUTPUT: dictionary, i.e {"HALLMARK_ADIPOGENESIS" : ["ABCA1","ABCB8", ...]}
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No such .gmt file: {path}")

    gene_sets: dict[str, list[str]] = {}

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            
            if not line.strip():
                continue
            parts = line.split("\t")

            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_num}: expected at least 3 tab-separated "
                    f"fields (name, description, genes...), got {len(parts)}"
                )

            name = parts[0]
            genes = [g for g in parts[2:] if g]
            gene_sets[name] = genes

    return gene_sets