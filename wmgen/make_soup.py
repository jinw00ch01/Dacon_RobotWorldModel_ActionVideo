"""Average adapter checkpoints of one training run into a single adapter file (CPU).

python -m wmgen.make_soup --out soup.pt --mode concat adapter_018000.pt adapter_020000.pt ...

mode "mean":   average every tensor key by key (LoRA A and B separately), rank unchanged.
mode "concat": exact average of the LoRA updates: B_i A_i / k summed by stacking ranks
               (A_cat = [A_1; ...; A_k], B_cat = [B_1/k, ..., B_k/k]); rank becomes k*r with alpha = rank,
               so the loaded scaling stays 1. The action embedder is averaged key by key in both modes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def soup(states: list[dict], mode: str) -> dict:
    k = len(states)
    keys = states[0]["lora"].keys()
    lora = {}
    for key in keys:
        ts = [s["lora"][key].double() for s in states]
        if mode == "mean":
            lora[key] = (sum(ts) / k).float()
        elif "lora_A" in key:
            lora[key] = torch.cat(ts, dim=0).float()
        elif "lora_B" in key:
            lora[key] = torch.cat([t / k for t in ts], dim=1).float()
        else:
            raise KeyError(key)
    embedder = {key: (sum(s["embedder"][key].double() for s in states) / k).float() for key in states[0]["embedder"]}
    args = dict(states[0]["args"])
    if mode == "concat":
        args["rank"] = int(args["rank"]) * k
    return {"lora": lora, "embedder": embedder, "step": [s["step"] for s in states], "args": args,
            "soup": {"mode": mode, "n": k}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["mean", "concat"], default="concat")
    args = ap.parse_args()
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in args.adapters]
    for s in states[1:]:
        assert s["lora"].keys() == states[0]["lora"].keys() and s["embedder"].keys() == states[0]["embedder"].keys()
        assert s["args"]["rank"] == states[0]["args"]["rank"]
    out = soup(states, args.mode)
    out["sources"] = [str(p) for p in args.adapters]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"{args.mode} soup of {len(states)} -> {args.out} rank {out['args']['rank']}")


if __name__ == "__main__":
    main()
