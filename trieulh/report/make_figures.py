"""Sinh toàn bộ biểu đồ cho báo cáo khoa học SLM 30M. Output: figures/*.png"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
FIG = os.path.join(D, "figures"); os.makedirs(FIG, exist_ok=True)
def load(n): return json.load(open(os.path.join(D, "data", n)))
BLUE, GRAY, GREEN, RED, ORANGE, CYAN = "#2563eb", "#94a3b8", "#16a34a", "#dc2626", "#d97706", "#0891b2"

a1 = load("analysis_30M.json"); a2 = load("analysis_30M_p2.json")
Lp2 = [h for h in load("loss_log_30M_p2.json") if "loss" in h]
steps = [h["step"] for h in Lp2]; losses = [h["loss"] for h in Lp2]
lrs = [h.get("learning_rate") for h in Lp2]; gn = [h.get("grad_norm") for h in Lp2]
PHASE = 1800

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

# 1. Training loss toàn cục (2 pha) + dải diễn giải
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhspan(2.0, max(losses)+0.3, color="#fecaca", alpha=.3); ax.axhspan(1.5, 2.0, color="#fef3c7", alpha=.4)
ax.axhspan(min(losses)-0.1, 1.5, color="#bbf7d0", alpha=.3)
ax.plot(steps, losses, color=BLUE, lw=1.6)
ax.axvline(PHASE, ls="--", color=RED, alpha=.7); ax.text(PHASE+30, max(losses), "Phase 2: corpus v2", color=RED, fontsize=9, va="top")
ax.axhline(1.8, ls=":", color=GRAY); ax.text(steps[-1], 1.8, " v1 baseline 1.8", color=GRAY, fontsize=8, va="bottom", ha="right")
ax.set_title("Training loss over steps (Phase 1 + Phase 2 resume)")
ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss"); ax.grid(alpha=.3)
ax.annotate(f"final {losses[-1]:.3f}", (steps[-1], losses[-1]), fontsize=9, color=BLUE)
save(fig, "01_loss_curve.png")

# 2. WSD learning-rate schedule
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(steps, lrs, color=ORANGE, lw=1.6); ax.axvline(PHASE, ls="--", color=RED, alpha=.5)
ax.set_title("Learning-rate schedule (Warmup-Stable-Decay, per phase)")
ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.grid(alpha=.3)
save(fig, "02_lr_schedule.png")

# 3. Gradient norm
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(steps, gn, color="#7c3aed", lw=1, alpha=.8); ax.axvline(PHASE, ls="--", color=RED, alpha=.5)
ax.set_title("Gradient norm (stability check; clipped at 1.0)")
ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.grid(alpha=.3)
save(fig, "03_grad_norm.png")

# 4. Scaling-law log-log fit
fig, ax = plt.subplots(figsize=(7, 5))
n0 = max(1, len(steps)//10); xs, ys = np.log(steps[n0:]), np.log(losses[n0:])
sl, ic = np.polyfit(xs, ys, 1); pred = sl*xs+ic
r2 = 1-((ys-pred)**2).sum()/((ys-ys.mean())**2).sum()
ax.loglog(steps, losses, color=BLUE, label="loss")
ax.loglog(np.exp(xs), np.exp(pred), ls="--", color=RED, label=f"fit: L ~ step^{sl:.2f} (R2={r2:.3f})")
ax.set_title("Scaling-law check (power-law regime)")
ax.set_xlabel("step (log)"); ax.set_ylabel("loss (log)"); ax.legend(); ax.grid(alpha=.3, which="both")
save(fig, "04_scaling_law.png")

# 5. Perplexity progression + theoretical floor
fig, ax = plt.subplots(figsize=(6.5, 4.5))
p1, p2 = a1.get("perplexity"), a2.get("perplexity")
fl1, fl2 = math.exp(a1["final_loss"]), math.exp(a2["final_loss"])
labels = ["Phase 1\n(1800 steps)", "Phase 2\n(3600 steps)"]
ax.bar(labels, [p1, p2], color=[GRAY, BLUE], width=.55)
for i,(v,fl) in enumerate([(p1,fl1),(p2,fl2)]):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.plot([i-.28,i+.28],[fl,fl], color=GREEN, lw=2)
ax.text(1.3, fl2, "e^loss floor", color=GREEN, fontsize=8, va="center")
ax.set_title("Held-out perplexity (lower is better)"); ax.set_ylabel("perplexity"); ax.grid(axis="y", alpha=.3)
save(fig, "05_perplexity.png")

# 6. Intrinsic quality vs real fables
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
q = a2["quality"]
m = ["distinct1","distinct2","self_bleu"]; g=[q[k]["gen"] for k in m]; r=[q[k]["real"] for k in m]
x=np.arange(len(m))
ax[0].bar(x-.2, g, .4, label="SLM 30M", color=BLUE); ax[0].bar(x+.2, r, .4, label="real fables", color=GRAY)
ax[0].set_xticks(x); ax[0].set_xticklabels(["Distinct-1","Distinct-2","Self-BLEU"]); ax[0].legend(); ax[0].grid(axis="y",alpha=.3)
ax[0].set_title("Diversity / repetition vs real")
ax[1].bar(["SLM 30M","real"], [q["flesch"]["gen"], q["flesch"]["real"]], color=[BLUE,GRAY], width=.5)
ax[1].axhspan(80,100,color="#bbf7d0",alpha=.4); ax[1].set_ylim(0,100)
ax[1].set_title("Flesch reading ease (80-100 = children band)"); ax[1].grid(axis="y",alpha=.3)
for i,v in enumerate([q["flesch"]["gen"],q["flesch"]["real"]]): ax[1].text(i,v,f"{v:.1f}",ha="center",va="bottom")
save(fig, "06_intrinsic_quality.png")

# 7. Story length distribution
fig, ax = plt.subplots(figsize=(7,4.2))
bins=np.linspace(min(a2["len_gen"]+a2["len_real"]), max(a2["len_gen"]+a2["len_real"]), 16)
ax.hist(a2["len_gen"], bins=bins, alpha=.6, label="SLM 30M", color=BLUE)
ax.hist(a2["len_real"], bins=bins, alpha=.6, label="real fables", color=GRAY)
ax.set_title(f"Story length (words), overlap {a2.get('len_overlap',0):.0%}"); ax.set_xlabel("words"); ax.legend(); ax.grid(alpha=.3)
save(fig, "07_length_dist.png")

# 8. Per-position loss
fig, ax = plt.subplots(figsize=(7,4))
pos=a2["pos_loss"]; xs=[i for i,v in enumerate(pos) if v is not None]; n=len(pos)
ax.plot([100*(i+.5)/n for i in xs],[pos[i] for i in xs], color=RED, marker="o", ms=4)
ax.set_title("Mean loss by story position"); ax.set_xlabel("position in story (%)"); ax.set_ylabel("cross-entropy"); ax.grid(alpha=.3)
save(fig, "08_position_loss.png")

# 9. Zipf
fig, ax = plt.subplots(figsize=(7,4.5))
zg,zr=a2["zipf_gen"],a2["zipf_real"]
ax.loglog(range(1,len(zg)+1), zg, color=BLUE, label="SLM 30M")
ax.loglog(range(1,len(zr)+1), zr, color=GRAY, label="real fables")
ax.set_title("Zipf: token rank vs frequency"); ax.set_xlabel("rank"); ax.set_ylabel("frequency"); ax.legend(); ax.grid(alpha=.3,which="both")
save(fig, "09_zipf.png")

# 10. Template collapse: owl rate
fig, ax = plt.subplots(figsize=(6.5,4.2))
ax.bar(["Phase 1","Phase 2\n(data fix)"], [0.90, a2["owl_rate_gen"]], color=[RED,GREEN], width=.5)
ax.axhline(0.28, ls="--", color=GRAY); ax.text(1.4,0.28,"real-data prior 28%",color=GRAY,fontsize=8,va="bottom",ha="right")
for i,v in enumerate([0.90,a2["owl_rate_gen"]]): ax.text(i,v,f"{v:.0%}",ha="center",va="bottom",fontweight="bold")
ax.set_title('Template collapse: "wise old owl" rate in generations'); ax.set_ylabel("fraction of stories"); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.3)
save(fig, "10_owl_rate.png")

# 11. Judge score progression (qualitative overall)
fig, ax = plt.subplots(figsize=(7.5,4.2))
stages=["v1\n(150k,900)","Phase1\n(rp1.3)","Phase1\n(rp1.1)","Phase2","Qwen-4B\n(ref)"]
scores=[2.5,6.0,6.2,7.0,9.75]; cols=[GRAY,CYAN,CYAN,BLUE,ORANGE]
ax.bar(stages,scores,color=cols,width=.6)
for i,v in enumerate(scores): ax.text(i,v,f"{v}",ha="center",va="bottom",fontweight="bold")
ax.set_title("Overall fable quality (LLM-judge, 1-10) across stages"); ax.set_ylabel("overall score"); ax.set_ylim(0,10.5); ax.grid(axis="y",alpha=.3)
save(fig, "11_score_progression.png")

# 12. Prompt-adherence: p2 -> dpo
fig, ax = plt.subplots(figsize=(6,4.2))
ax.bar(["Phase 2","Phase 2 + DPO"], [0.71, 0.76], color=[BLUE,GREEN], width=.5)
for i,v in enumerate([0.71,0.76]): ax.text(i,v,f"{v:.0%}",ha="center",va="bottom",fontweight="bold")
ax.set_title("Prompt-adherence (slot recall) after DPO alignment"); ax.set_ylabel("slot recall"); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.3)
save(fig, "12_adherence_dpo.png")

# 13. DPO training dynamics (từ trial log)
fig, ax = plt.subplots(1,2,figsize=(11,3.8))
st=[5,10,15,20,25,30]; acc=[.475,.6,.509,.975,1.0,1.0]; marg=[-0.0027,0.0038,-0.0023,0.1944,0.2082,0.215]; loss=[.6945,.6912,.6947,.6039,.5961,.591]
ax[0].plot(st,acc,marker="o",color=GREEN,label="reward accuracy"); ax[0].plot(st,loss,marker="s",color=BLUE,label="DPO loss")
ax[0].set_title("DPO: reward accuracy + loss"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(st,marg,marker="o",color=ORANGE); ax[1].axhline(0,ls=":",color=GRAY)
ax[1].set_title("DPO: reward margin (chosen - rejected)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("margin"); ax[1].grid(alpha=.3)
save(fig, "13_dpo_dynamics.png")

# 14. Speed comparison (inference tok/s)
fig, ax = plt.subplots(figsize=(6,4.2))
ax.bar(["SLM 30M","Qwen-4B"], [949, 19], color=[BLUE,ORANGE], width=.5, log=True)
for i,v in enumerate([949,19]): ax.text(i,v,f"{v} tok/s",ha="center",va="bottom",fontweight="bold")
ax.set_title("Inference speed (~50x faster, 130x smaller)"); ax.set_ylabel("tokens/sec (log)"); ax.grid(axis="y",alpha=.3,which="both")
save(fig, "14_speed.png")

# 15. Free vs conditioned generation
try:
    fc = load("free_vs_conditioned.json"); F, C = fc["free"], fc["cond"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    m = ["d1", "d2", "sb"]; labels = ["Distinct-1", "Distinct-2", "Self-BLEU"]
    x = np.arange(len(m))
    ax[0].bar(x-.2, [F[k] for k in m], .4, label="free", color=GRAY)
    ax[0].bar(x+.2, [C[k] for k in m], .4, label="5-slot conditioned", color=BLUE)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
    ax[0].set_title("Diversity: conditioning raises Distinct, lowers Self-BLEU")
    ax[1].bar(["Flesch\n(free)","Flesch\n(cond)","Slot recall\n(cond)"],
              [F["flesch"], C["flesch"], C["slot_recall"]*100],
              color=[GRAY, BLUE, GREEN])
    ax[1].axhspan(80,100,color="#bbf7d0",alpha=.3); ax[1].set_ylim(0,100)
    for i,v in enumerate([F["flesch"],C["flesch"],C["slot_recall"]*100]): ax[1].text(i,v,f"{v:.0f}",ha="center",va="bottom")
    ax[1].set_title("Readability + adherence (conditioned)"); ax[1].grid(axis="y", alpha=.3)
    save(fig, "15_free_vs_conditioned.png")
except FileNotFoundError:
    print("skip fig 15 (no data)")

print("DONE:", len(os.listdir(FIG)), "figures")

# ---- Fig 16: post-training campaign summary (4 methods null vs best-of-N win) ----
try:
    camp = load("posttraining_campaign.json")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    exps = camp["experiments"] + [camp["best_of_n"]]
    x = np.arange(len(exps))
    b = [e["baseline"] for e in exps]; m = [e["method"] for e in exps]
    ax.bar(x-.2, b, .4, label="baseline (30M-p2)", color=GRAY)
    colors = [ORANGE]*len(camp["experiments"]) + [GREEN]
    ax.bar(x+.2, m, .4, label="method", color=colors)
    for i, e in enumerate(exps):
        d = e["method"]-e["baseline"]
        ax.text(i+.2, e["method"]+.06, f"{d:+.2f}\n(n={e['n']})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([e["name"] for e in exps], fontsize=8.5)
    ax.set_ylim(6.5, 9.3); ax.set_ylabel("LLM-judge overall (held-out)")
    ax.axhline(8.55, ls=":", color=GREEN, alpha=.5)
    ax.set_title("Post-training campaign: four training methods are null; inference-time best-of-N is the only confirmed gain")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=.3)
    save(fig, "16_posttraining_campaign.png")
except FileNotFoundError:
    print("skip fig 16 (no data)")

# ---- Fig 17: GRPO training dynamics (reward curve + KL) ----
try:
    rows = [json.loads(l) for l in open(os.path.join(D, "data", "grpo_log.jsonl"))]
    st = [r["step"] for r in rows]; rw = [r["reward_mean"] for r in rows]
    kl = [abs(r["kl"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(st, rw, "o-", ms=3, color=BLUE, alpha=.7, label="reward mean (16 rollouts)")
    w = 5
    ma = [np.mean(rw[max(0,i-w+1):i+1]) for i in range(len(rw))]
    ax[0].plot(st, ma, color=ORANGE, lw=2, label=f"moving avg ({w})")
    ax[0].axvline(30.5, ls="--", color=GRAY); ax[0].text(31, min(rw)+.05, "lr 3e-6 -> 1e-5", fontsize=8)
    ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("judge reward"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("In-training reward stays flat (judge noise dominates)")
    ax[1].semilogy(st, kl, "s-", ms=3, color=BLUE)
    ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("|KL(policy || ref)| nats/token (log)")
    ax[1].axvline(30.5, ls="--", color=GRAY)
    ax[1].set_title("Policy shift stays tiny (KL ~ 1e-3 at end)")
    ax[1].grid(alpha=.3, which="both")
    save(fig, "17_grpo_dynamics.png")
except FileNotFoundError:
    print("skip fig 17 (no data)")

# ---- Fig 18: judge measurement noise (the methodological lesson) ----
try:
    nz = load("posttraining_campaign.json")["noise"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    groups = [("30M-p2\n(same model, 2 runs)", nz["p2"], GRAY),
              ("30M-raft\n(same model, 2 runs)", nz["raft"], ORANGE),
              ("30M-grpo\n(n=15, 2 runs)", nz["grpo_n15"], BLUE)]
    for i, (name, vals, c) in enumerate(groups):
        ax.scatter([i]*len(vals), vals, s=90, color=c, zorder=3)
        ax.plot([i, i], [min(vals), max(vals)], color=c, lw=2, alpha=.5)
        for v in vals:
            ax.text(i+.08, v, f"{v:.2f}", va="center", fontsize=9)
    ax.scatter([3], [nz["grpo_n45"]], s=110, color=GREEN, zorder=3, marker="D")
    ax.text(3.08, nz["grpo_n45"], f'{nz["grpo_n45"]:.2f}', va="center", fontsize=9)
    ax.annotate("n=15 spread collapses at n=45", xy=(3, nz["grpo_n45"]), xytext=(1.9, 8.75),
                arrowprops=dict(arrowstyle="->", color=GREEN), fontsize=9, color=GREEN)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups] + ["30M-grpo\n(n=45)"], fontsize=8.5)
    ax.set_ylabel("LLM-judge overall")
    ax.set_title("Repeated measurements of the SAME model differ by up to 0.45:\njudge noise bounds what n=15 evals can conclude")
    ax.grid(axis="y", alpha=.3)
    save(fig, "18_judge_noise.png")
except FileNotFoundError:
    print("skip fig 18 (no data)")

print("EXTRA FIGS DONE")

# ---- Fig 19: head-to-head p1/p2/dpo x 2 use case (judge + Claude) ----
try:
    hh = load("headtohead_summary.json")
    MODELS = ["slm-30m", "slm-30m-p2", "slm-30m-dpo", "slm-60m"]
    LBL = ["Phase 1", "Phase 2", "Phase 2+DPO", "60M"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(MODELS))
    for ax, uc, title in [(axes[0], "UC1_free", "UC1: free generation (no slots, short)"),
                          (axes[1], "UC2_slots", "UC2: 5-slot conditioned (short)")]:
        j = [hh["judge_means"][uc][m] for m in MODELS]
        c = [hh["claude_means"][uc][m] for m in MODELS]
        ax.bar(x - .2, j, .4, label="LLM-judge (qwen3-4b)", color=BLUE)
        ax.bar(x + .2, c, .4, label="Claude (manual read)", color=ORANGE)
        for i in range(len(MODELS)):
            ax.text(i - .2, j[i] + .06, f"{j[i]:.2f}", ha="center", fontsize=8)
            ax.text(i + .2, c[i] + .06, f"{c[i]:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=9)
        ax.set_ylim(0, 10); ax.grid(axis="y", alpha=.3)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("overall score (/10)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Head-to-head, paired seeds, n=4/cell: gains come from pretraining (P1->P2->60M), not from DPO", fontsize=10)
    save(fig, "19_headtohead_progression.png")
except FileNotFoundError:
    print("skip fig 19 (no data)")
