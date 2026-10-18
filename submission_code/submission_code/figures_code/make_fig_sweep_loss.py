import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, textwrap, csv, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BL,OR="#2a78d6","#eb6834"; INK="#222222"; MUTE="#555555"; LINE="#9a9a9a"; CARD="#f5f5f5"; BAND="#cfe0f7"
R={r["run"]:r for r in csv.DictReader(open("sweep_run_summary.csv"))}
def g(m,k):
    r=R.get(f"{m}__{k}"); return float(r["best_eval"]) if r and r["best_eval"] else np.nan
GR,QW="granite-4.1-8b","qwen3-8b"
# baseline (attn+MLP, r16, alpha32, lr3e-4, 3 epochs) rows from the modules stage
BASE={GR:1.0533784627914429,QW:1.1358513832092285}
def stage(keys,base_at=None):
    return {m:[(BASE[m] if k=="BASE" else g(m,k)) for k in keys] for m in (GR,QW)}
S=[("1  Rank",["8","16","32","64"],stage(["lora_r8","lora_r16","lora_r32","lora_r64"]),"16",
    "Flat: within 0.006. Rank 64: Qwen out of memory."),
   ("2  Learning rate",["5e-5","1e-4","2e-4","3e-4","5e-4","7e-4","1e-3"],stage(["lr5e-05","lr0.0001","lr0.0002","lr0.0003","lr0.0005","lr0.0007","lr0.001"]),"3e-4",
    "3e-4 beats 2e-4 by 0.006 / 0.018, at the best checkpoint only. Granite diverged at 1e-3."),
   ("3  Target modules",["attention\nonly","attention\n+ MLP"],stage(["attn_onlyTrue","BASE"]),"attention\n+ MLP","Adding the MLP lowers loss by about 0.07."),
   ("4  Alpha",["16","32","64"],stage(["lora_alpha16","BASE","lora_alpha64"]),"32","Differences at most 0.03; kept 32."),
   ("5  Epochs",["2","3","4"],stage(["epochs2","BASE","epochs4"]),"3","All arms scored at about epoch 1.9: not a test of training length.")]
W,H=13.33,7.5
f=plt.figure(figsize=(W,H)); bg=f.add_axes([0,0,1,1]); bg.set_xlim(0,W); bg.set_ylim(0,H); bg.axis("off")
def rb(x,y,w,h,fc,ec,lw=1.2,r=0.12,ls="-"): bg.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls))
bg.text(W/2,H-0.38,"How each hyperparameter was picked: held-out loss at each stage",ha="center",va="center",fontsize=22,color=INK,fontweight="bold")
lx=1.0
bg.plot([lx],[H-0.86],"o",color=BL,ms=8); bg.text(lx+0.15,H-0.86,"Granite-4.1-8B",fontsize=13,va="center",color=INK)
bg.plot([lx+2.0],[H-0.86],"s",color=OR,ms=8); bg.text(lx+2.15,H-0.86,"Qwen3-8B",fontsize=13,va="center",color=INK)
bg.text(lx+3.4,H-0.86,"y = held-out loss at the best checkpoint (about epoch 1.9)",fontsize=13,va="center",color=MUTE)
rb(lx+9.0,H-0.99,0.35,0.26,BAND,BAND,r=0.03); bg.text(lx+9.45,H-0.86,"value carried forward",fontsize=13,va="center",color=MUTE)
widths=[2.15,3.3,2.15,2.15,2.2]; gap=0.25; x=0.2; top=H-1.2; ch=4.95; cy=top-ch
YL=(1.04,1.26)
for k,((name,vals,d,chosen,note),cw) in enumerate(zip(S,widths)):
    rb(x,cy,cw,ch,CARD,LINE)
    bg.text(x+cw/2,top-0.33,name,ha="center",va="center",fontsize=15,color=INK,fontweight="bold")
    axw=cw-0.75; ax=f.add_axes([(x+0.62)/W,(cy+1.5)/H,axw/W,2.8/H])
    n=len(vals); xs=np.arange(n)
    ci=vals.index(chosen); ax.axvspan(ci-.45,ci+.45,color=BAND,lw=0)
    for m,c,mk in ((GR,BL,"o"),(QW,OR,"s")):
        y=np.array(d[m],float); yc=np.clip(y,YL[0],YL[1]+0.0)
        ax.plot(xs,y,"-"+mk,color=c,lw=2.2,ms=6.5,clip_on=True)
    ax.set_ylim(*YL); ax.set_xlim(-.5,n-.5); ax.set_yticks([1.05,1.10,1.15,1.20,1.25])
    ax.set_xticks(xs); ax.set_xticklabels(vals,fontsize=10.5 if n>4 else 11.5)
    for t in ax.get_xticklabels():
        if t.get_text()==chosen: t.set_color(BL); t.set_fontweight("bold")
    ax.tick_params(axis="y",labelsize=10.5,length=0); ax.tick_params(axis="x",length=0)
    ax.grid(axis="y",color="#dddddd",lw=.8); ax.set_axisbelow(True)
    for s in("top","right","left"): ax.spines[s].set_visible(False)
    if k==1:
        ax.text(5.12,1.205,"7e-4: 1.34\n(off scale)",ha="left",va="center",fontsize=9,color=BL)
        ax.plot([5],[1.255],"^",color=BL,ms=6)
        ax.text(6,1.245,"✗",color=BL,fontsize=15,ha="center",va="center",fontweight="bold")
        ax.text(6,1.228,"diverged",color=BL,fontsize=8.5,ha="center",va="center")
    if k==0:
        ax.text(3,1.16,"Qwen\nOOM",color=OR,fontsize=9.5,ha="center",va="center")
    bg.text(x+cw/2,cy+0.1,"\n".join(textwrap.wrap(note,int(cw*10.5))),ha="center",va="bottom",fontsize=12.5,color=MUTE,linespacing=1.25)
    if k<4: bg.add_patch(FancyArrowPatch((x+cw+0.02,cy+ch/2),(x+cw+gap-0.02,cy+ch/2),arrowstyle="-|>",mutation_scale=18,lw=2.5,color="#666666"))
    x+=cw+gap
bg.text(0.2+0.0,cy+ch+0.0,"",fontsize=1)
rb(0.2,0.2,W-0.4,0.8,"white","#666666",lw=1.2,ls=(0,(4,3)))
bg.text(W/2,0.6,"32 runs, 1 GPU each. Then: production adapters at 1,000 / 2,000 / 3,800 examples,\nplus two extra learning-rate arms (2e-4, 5e-5) at 3,800.",ha="center",va="center",fontsize=14,color=INK,linespacing=1.4)
f.savefig("figs/fig_sweep_loss_flow_slide.png",dpi=200); print("saved")
