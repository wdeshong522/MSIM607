import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
from matplotlib.patches import Patch
rows=list(csv.DictReader(open("paper50_results_long.csv")))
def get(j,cond,m): return float(next(r for r in rows if r["judge"]==j and r["condition"]==cond and r["metric"]==m)["mean"])
keys=["correctness","completeness","faithfulness","relevance","instruction_following","safety","hallucination"]
names=["Correct.","Complete.","Faithful.","Relevance","Instr. following","Safety"]
CONDS={"Granite-4.1-8B":"Granite / tuned n=3,800 (lr 3e-4)","Qwen3-8B":"Qwen3 / tuned n=3,800 (lr 3e-4)"}
COL={"Granite-4.1-8B":"#2a78d6","Qwen3-8B":"#eb6834"}
def fig_judge(jd,path,size,fs,dec,lab_fs):
    D={m:[get(jd,c,k) for k in keys] for m,c in CONDS.items()}
    f,(a1,a2)=plt.subplots(1,2,figsize=size,gridspec_kw={"width_ratios":[6,1.1],"wspace":.08},constrained_layout=True)
    models=list(CONDS); w=.34
    for ax,idx,nm in ((a1,list(range(6)),names),(a2,[6],["Halluc.\n(flag rate)"])):
        x=np.arange(len(idx))
        for j,m in enumerate(models):
            v=[D[m][i] for i in idx]; xs=x+(j-.5)*w
            ax.bar(xs,v,w*.92,color=COL[m],edgecolor="white",linewidth=0)
            for xi,vi in zip(xs,v): ax.text(xi,vi+.01,f"{vi:.{dec}f}",ha="center",va="bottom",fontsize=lab_fs,color="#333")
        ax.set_xticks(x); ax.set_xticklabels([n.replace("Instr. following","Instr.\nfollowing") for n in nm],fontsize=fs)
        for s in("top","right"): ax.spines[s].set_visible(False)
        ax.grid(axis="y",color="#e6e6e6",lw=.8); ax.set_axisbelow(True); ax.tick_params(labelsize=fs)
    a1.set_ylim(0,1.1); a2.set_ylim(0,.6); a2.set_yticks([0,.2,.4,.6])
    a1.set_ylabel("Mean score (0 to 1)",fontsize=fs); a1.set_title(f"{jd} judge: quality metrics (higher is better)",fontsize=fs+1); a2.set_title("Lower is\nbetter",fontsize=fs+1)
    a1.legend(handles=[Patch(color=COL[m],label=m) for m in models],ncol=2,fontsize=fs-1,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.2))
    f.savefig(path,dpi=200,bbox_inches="tight"); plt.close(f)
    return D
for jd,tag in (("Qwen3.8-27B","qwenjudge"),("Gemma-4-31B-it","gemmajudge")):
    D=fig_judge(jd,f"figs/fig_metrics50_{tag}.png",(5.4,3.0),8,3,5)
    fig_judge(jd,f"figs/fig_metrics50_{tag}_slide.png",(11.5,5.8),15,3,10.5)
    print(jd); [print(" ",m,[round(v,3) for v in D[m]]) for m in D]
