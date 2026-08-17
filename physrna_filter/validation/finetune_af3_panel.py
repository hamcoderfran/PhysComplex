"""
Contrastive fine-tuning of the PhysGT interface head on AF3 positive/negative pairs.

Uses the holdout evaluation panel (P1–P5 vs N1–N5) to teach the model that
AF3 wrong-partner structures should score worse than true pairs.

Example
-------
    # After train_interface_head + merge_gt_checkpoint:
    python -m physrna_filter.validation.finetune_af3_panel \\
        --zips ./af3_predictions \\
        --epochs 50 \\
        --checkpoint physrna_filter/validation/gt_checkpoint.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ..analysis.gt_inference import (
    _graph_to_device,
    build_af3_graph,
    load_gt_model,
)
from ..analysis.physics_edge import coerce_edge_attr
from ..data.af3_eval_panel import (
    contrastive_pairs,
    filter_contrastive_pairs,
    load_af3_eval_panel,
    panel_zip_path,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_panel_graph(zip_path: Path, model_rank: int = 0, meta: dict | None = None):
    return build_af3_graph(
        str(zip_path),
        esm_dim=meta.get("esm_dim") if meta else None,
        rnafm_dim=meta.get("rnafm_dim") if meta else None,
        edge_dim=meta.get("edge_dim") if meta else None,
        model_rank=model_rank,
    )


def finetune_af3_panel(
    zip_dir: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 50,
    lr: float = 1e-4,
    margin: float = 1.0,
    seed: int = 7,
    model_rank: int = 0,
    panel_path: str | Path | None = None,
    partner_groups: list[str] | None = None,
    entry_ids: list[str] | None = None,
) -> Path:
    random.seed(seed)
    torch.manual_seed(seed)

    zip_dir = Path(zip_dir)
    checkpoint_path = Path(checkpoint_path)
    panel = load_af3_eval_panel(panel_path)

    pairs: list[tuple[Path, Path]] = []
    raw_pairs = filter_contrastive_pairs(
        contrastive_pairs(panel),
        partner_groups=partner_groups,
        entry_ids=entry_ids,
    )
    for pos_entry, neg_entry in raw_pairs:
        pos_zip = panel_zip_path(pos_entry, zip_dir)
        neg_zip = panel_zip_path(neg_entry, zip_dir)
        if pos_zip is None or neg_zip is None:
            print(
                f"  SKIP pair {pos_entry['id']}/{neg_entry['id']}: "
                f"zip missing under {zip_dir}"
            )
            continue
        pairs.append((pos_zip, neg_zip))

    if not pairs:
        scope = ""
        if partner_groups:
            scope += f" partner_groups={partner_groups}"
        if entry_ids:
            scope += f" entry_ids={entry_ids}"
        raise RuntimeError(
            f"No AF3 zip pairs found in {zip_dir}.{scope} "
            "Download Server zips matching af3_eval_panel.json names."
        )

    scope_msg = ""
    if partner_groups:
        scope_msg = f" [{', '.join(partner_groups)}]"
    print(f"AF3 panel fine-tune: {len(pairs)} contrastive pair(s){scope_msg}, {epochs} epochs")
    model, meta = load_gt_model(str(checkpoint_path))
    model = model.to(DEVICE)
    edge_dim = meta.get("edge_dim", getattr(model, "edge_dim", 15))

    for name, param in model.named_parameters():
        param.requires_grad = any(
            tag in name for tag in ("interface_head", "cross_attn", "edge_gates", "physics_bias")
        )

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    graph_cache: dict[str, object] = {}

    def get_graph(zip_path: Path):
        key = str(zip_path.resolve())
        if key not in graph_cache:
            graph_cache[key] = _load_panel_graph(zip_path, model_rank=model_rank, meta=meta)
        return graph_cache[key]

    model.train()
    for epoch in range(1, epochs + 1):
        order = list(range(len(pairs)))
        random.shuffle(order)
        total_loss = 0.0

        for idx in order:
            pos_zip, neg_zip = pairs[idx]
            pos_g = get_graph(pos_zip)
            neg_g = get_graph(neg_zip)
            pos = _graph_to_device(pos_g)
            neg = _graph_to_device(neg_g)

            pos_edge = coerce_edge_attr(pos.edge_attr, edge_dim)
            neg_edge = coerce_edge_attr(neg.edge_attr, edge_dim)
            s_pos = model.score_interface(
                pos.x_protein, pos.x_rna, pos.edge_index, pos_edge,
                pos.x_protein.shape[0],
            )
            s_neg = model.score_interface(
                neg.x_protein, neg.x_rna, neg.edge_index, neg_edge,
                neg.x_protein.shape[0],
            )
            loss = F.relu(margin + s_pos - s_neg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            total_loss += float(loss.detach())

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  loss={total_loss / len(pairs):.4f}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from ..config import backup_checkpoint

    bak = backup_checkpoint(checkpoint_path)
    if bak:
        print(f"  Backup saved to {bak}")
    payload["model_state"] = model.state_dict()
    payload["interface_head_trained"] = True
    payload["deployed"] = True
    payload["af3_panel_finetuned"] = True
    if partner_groups:
        payload["af3_panel_partner_groups"] = list(partner_groups)
    torch.save(payload, checkpoint_path)
    print(f"Saved AF3-panel fine-tuned checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune interface head on AF3 P/N panel")
    ap.add_argument(
        "--zips",
        default="af3_predictions",
        help="Directory with fold_p1_*.zip … fold_n5_*.zip",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to fine-tune (default: ~/.physrna/gt_checkpoint.pt)",
    )
    ap.add_argument("--panel", default=None, help="Path to af3_eval_panel.json")
    ap.add_argument(
        "--partner-groups",
        nargs="+",
        default=None,
        help="Fine-tune only these partner groups (e.g. ms2 lin28)",
    )
    ap.add_argument(
        "--entry-ids",
        nargs="+",
        default=None,
        help="Fine-tune only pairs touching these panel ids (e.g. P2 N2)",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-rank", type=int, default=0)
    args = ap.parse_args()

    from ..config import writable_gt_checkpoint

    finetune_af3_panel(
        zip_dir=args.zips,
        checkpoint_path=writable_gt_checkpoint(args.checkpoint),
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        seed=args.seed,
        model_rank=args.model_rank,
        panel_path=args.panel,
        partner_groups=args.partner_groups,
        entry_ids=args.entry_ids,
    )


if __name__ == "__main__":
    main()
