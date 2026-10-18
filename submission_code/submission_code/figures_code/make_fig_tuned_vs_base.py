import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
from matplotlib.patches import Patch
rows=list(csv.DictReader(open("paper50_results_long.csv")))
def get(j,cond,m):
    r=next(r for r in rows if r["judge"]==j and r["condition"]==cond and r["metric"]==m)
    return float(r["mean"]),float(r["ci_lo"]),float(r["ci_hi"]),int(r["n_scored"])
M=[("correctness","Correct."),("completeness","Complete."),("faithfulness","Faithful."),("relevance","Relevance"),("instruction_following","Instr.\nfollowing"),("safety","Safety")]
COND={"Granite-4.1-8B":("Granite / base (untuned)","Granite / tuned n=3,800 (lr 3e-4)","#2a78d6"),
      "Qwen3-8B":("Qwen3 / base (untuned)","Qwen3 / tuned n=3,800 (lr 3e-4)","#eb6834")}
GREY="#b8b8b8"
def fig(judge,tag,size,fs,lab):
    f,axs=plt.subplots(2,2,figsize=size,gridspec_kw={"width_ratios":[6,1.1],"wspace":.08},constrained_layout=True)
    f.get_layout_engine().set(hspace=0.12,h_pad=0.05)
    for r,(model,(b,t,col)) in enumerate(COND.items()):
        for c,(ax,keys) in enumerate(((axs[r,0],M),(axs[r,1],[("hallucination","Halluc.\n(flag rate)")]))):
            x=np.arange(len(keys)); w=.36
            for k,(cond,color) in enumerate(((b,GREY),(t,col))):
                v=[get(judge,cond,m) for m,_ in keys]; xs=x+(k-.5)*w
                mean=np.array([a[0] for a in v]); lo=mean-np.array([a[1] for a in v]); hi=np.array([a[2] for a in v])-mean
                ax.bar(xs,mean,w*.92,color=color,edgecolor="white",linewidth=0)
                if lab:
                    for xi,a in zip(xs,v): ax.text(xi,a[0]+.012,f"{a[0]:.3f}",ha="center",va="bottom",fontsize=lab,color="#333")
            ax.set_xticks(x); ax.set_xticklabels([n for _,n in keys],fontsize=fs)
            for s in("top","right"): ax.spines[s].set_visible(False)
            ax.grid(axis="y",color="#e6e6e6",lw=.8); ax.set_axisbelow(True); ax.tick_params(labelsize=fs)
            if c==0: ax.set_ylim(0,1.15); ax.set_ylabel("Mean score (0 to 1)",fontsize=fs); ax.set_title(f"{model}: quality metrics (higher is better)",fontsize=fs+1,loc="left")
            else: ax.set_ylim(0,.8); ax.set_yticks([0,.2,.4,.6,.8]); ax.set_title("Lower is better",fontsize=fs+1)
    h=[Patch(color=GREY,label="Untuned base"),Patch(color="#555",label="Tuned (3,800 examples, lr 3e-4)")]
    axs[1,0].legend(handles=h,ncol=2,fontsize=fs,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.38))
    f.suptitle(f"{judge} judge (mean scores, 50 held-out prompts)",fontsize=fs+2)
    f.savefig(f"figs/fig_tuned_vs_base_{tag}.png",dpi=200,bbox_inches="tight"); plt.close(f)
for j,tag in (("Qwen3.8-27B","qwenjudge"),("Gemma-4-31B-it","gemmajudge")):
    fig(j,tag,(7.2,4.1),8,5.5)
    # slide version
    import matplotlib as mpl
    fig_size=(12.5,6.4)
    f=fig(j,tag+"_slide",fig_size,13,10)
