import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, csv, numpy as np, os
exec(open("make_fig_sweep_loss.py").read().split("W,H=13.33,7.5")[0].split("import matplotlib; matplotlib.use(\"Agg\")")[1].replace("import matplotlib.pyplot as plt, textwrap, csv, numpy as np\n","").replace("from matplotlib.patches import FancyBboxPatch, FancyArrowPatch",""))
os.makedirs("figs/sweep_panels",exist_ok=True)
YL=(1.04,1.26)
sizes=[4.0,6.0,4.0,4.0,4.0]; H=4.2
for k,((name,vals,d,chosen,note),w) in enumerate(zip(S,sizes),1):
    f=plt.figure(figsize=(w,H)); ax=f.add_axes([0.24 if w<5 else 0.15,0.17,0.73 if w<5 else 0.83,0.68])
    n=len(vals); xs=np.arange(n); ci=vals.index(chosen)
    ax.axvspan(ci-.45,ci+.45,color=BAND,lw=0)
    for m,c,mk in ((GR,BL,"o"),(QW,OR,"s")):
        ax.plot(xs,d[m],"-"+mk,color=c,lw=2.8,ms=8,clip_on=True)
    ax.set_ylim(*YL); ax.set_xlim(-.5,n-.5); ax.set_yticks([1.05,1.10,1.15,1.20,1.25]); ax.set_xticks(xs); ax.set_xticklabels(vals,fontsize=14)
    for t in ax.get_xticklabels():
        if t.get_text()==chosen: t.set_color(BL); t.set_fontweight("bold")
    ax.tick_params(axis="y",labelsize=13,length=0); ax.tick_params(axis="x",length=0)
    ax.grid(axis="y",color="#dddddd",lw=.8); ax.set_axisbelow(True)
    for s in("top","right","left"): ax.spines[s].set_visible(False)
    ax.set_ylabel("Held-out loss",fontsize=13)
    f.text(0.5,0.93,name,ha="center",va="center",fontsize=19,fontweight="bold",color=INK)
    if k==1: ax.text(3,1.16,"Qwen\nOOM",color=OR,fontsize=11,ha="center",va="center")
    if k==2:
        ax.plot([5],[1.255],"^",color=BL,ms=8); ax.text(5.15,1.19,"7e-4: 1.34\n(off scale)",ha="left",va="center",fontsize=11,color=BL)
        ax.text(6,1.247,"✗",color=BL,fontsize=18,ha="center",va="center",fontweight="bold"); ax.text(6,1.228,"diverged",color=BL,fontsize=11,ha="center",va="center")
    out=f"figs/sweep_panels/sweep_{k}_{name.split('  ')[1].lower().replace(' ','_')}.png"
    f.savefig(out,dpi=250,transparent=True); plt.close(f); print(out)
# legend strip
f=plt.figure(figsize=(7.5,0.5)); a=f.add_axes([0,0,1,1]); a.axis("off"); a.set_xlim(0,7.5); a.set_ylim(0,1)
a.plot([0.25],[.5],"o",color=BL,ms=10); a.text(0.45,.5,"Granite-4.1-8B",fontsize=14,va="center",color=INK)
a.plot([2.75],[.5],"s",color=OR,ms=10); a.text(2.95,.5,"Qwen3-8B",fontsize=14,va="center",color=INK)
a.add_patch(plt.Rectangle((4.45,.25),.4,.5,color=BAND,lw=0)); a.text(5.0,.5,"value carried forward",fontsize=14,va="center",color=INK)
f.savefig("figs/sweep_panels/sweep_legend.png",dpi=250,transparent=True)
