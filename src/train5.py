# train5.py
from __future__ import annotations
import math
from torch.utils.data import DataLoader
import torch

from config5 import DEVICE, CHECKPOINT_DIR, EPOCHS, LR, PRINT_EVERY, SAVE_EVERY
from data5 import GraphStructureDataset5
from models5 import StructureToGraphDecoder5
from losses5 import LossPack5

def train_loop():
    ds = GraphStructureDataset5(data_dir=__import__("config5").DATA_DIR)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    model = StructureToGraphDecoder5().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    loss_pack = LossPack5()
    best_bce = float("inf")

    for epoch in range(1, EPOCHS + 1):
        steps, sum_loss = 0, 0.0
        sum_terms = {}

        for batch in loader:
            s_vec = batch["s_vec"].to(DEVICE)          # [5]
            A_tgt = batch["A_target"].to(DEVICE)       # [N,N]
            T_tgt = batch["T_target"].to(DEVICE)       # [N,N]
            widths = batch["meta"]["widths"]           # list[int]

            # 用 s_vec 里的 total_T（反归一化）
            from config5 import NORM_T
            target_total_T = (s_vec[4] * NORM_T).item()

            model.train()
            A_logits, T_pred, _ = model(s_vec, widths=widths)
            loss, terms = loss_pack(A_logits, A_tgt, T_pred, T_tgt, target_total_T, widths)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            steps += 1
            sum_loss += float(loss.detach().item())
            for k, v in terms.items():
                sum_terms[k] = sum_terms.get(k, 0.0) + float(v)

        if steps == 0:
            print(f"[Epoch {epoch:03d}] skipped (no data)")
            continue

        avg_terms = {k: v/steps for k, v in sum_terms.items()}
        # 用loss权重重算一个可读的显示
        from config5 import (W_BCE, W_TIME, W_TOTALT, W_LONGEST, W_DAG, W_DEG_COV, W_SRC_SINK_SOFT, W_TIME_NODE)
        avg_loss_disp = (W_BCE*avg_terms.get("bce",0.0)
                         + W_TIME*avg_terms.get("time",0.0)
                         + W_TOTALT*avg_terms.get("totalT",0.0)
                         + W_DEG_COV*avg_terms.get("deg_cov",0.0)
                         + W_SRC_SINK_SOFT*avg_terms.get("src_sink",0.0)
                         + W_LONGEST*avg_terms.get("longest",0.0)
                         + W_DAG*avg_terms.get("dag_h",0.0)
                         + W_TIME_NODE * avg_terms.get("time_node", 0.0)  # ★ 新增
                         )

        if epoch % PRINT_EVERY == 0:
            keys = ["bce","deg_cov","src_sink","time","time_node","totalT","longest","dag_h"]
            terms_str = ", ".join(f"{k}:{avg_terms.get(k,0.0):.4f}" for k in keys)
            print(f"[Epoch {epoch:03d}] loss={avg_loss_disp:.4f} | {terms_str}")

        # 保存 best（按 bce）
        if math.isfinite(avg_terms.get("bce", float("inf"))) and avg_terms["bce"] < best_bce:
            best_bce = avg_terms["bce"]
            ckpt_best = CHECKPOINT_DIR / "decoder_best1.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_best)
            print(f"Saved BEST checkpoint (bce={best_bce:.4f}) to {ckpt_best}")

        # 阶段性备份
        if SAVE_EVERY and epoch % SAVE_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"decoder_epoch{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    ckpt_path = CHECKPOINT_DIR / "decoder_final1.pt"
    torch.save({"model": model.state_dict(), "epoch": EPOCHS}, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train_loop()
