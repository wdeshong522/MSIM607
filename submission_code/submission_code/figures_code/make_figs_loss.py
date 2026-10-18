import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, csv, sys
B,O,G="#2a78d6","#eb6834","#1aa37a"
def style(ax,fs):
    for s in("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#e6e6e6",lw=.8); ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs)
# ---- LR sweep
rows={r["run"]:r for r in csv.DictReader(open("sweep_run_summary.csv"))}
lrs=[("5e-05",5e-5),("0.0001",1e-4),("0.0002",2e-4),("0.0003",3e-4),("0.0005",5e-4),("0.0007",7e-4),("0.001",1e-3)]
def series(m):
    xs,b,l=[],[],[]
    for k,v in lrs:
        r=rows.get(f"{m}__lr{k}")
        if r and r["best_eval"]:
            xs.append(v);b.append(float(r["best_eval"]));l.append(float(r["last_eval"]))
    return xs,b,l
def fig_lr(path,size,fs):
    f,axs=plt.subplots(1,2,figsize=size,constrained_layout=True)
    for ax,m,t in zip(axs,("granite-4.1-8b","qwen3-8b"),("Granite-4.1-8B","Qwen3-8B")):
        x,b,l=series(m); style(ax,fs)
        ax.plot(x,b,"-o",color=B,lw=2.4,ms=7); ax.plot(x,l,"--s",color=O,lw=2.4,ms=7)
        ax.set_xscale("log"); tk=[5e-5,1e-4,2e-4,3e-4,5e-4,1e-3]; ax.set_xticks(tk); ax.set_xticklabels(["5e-5","1e-4","2e-4","3e-4","5e-4","1e-3\n(diverged)" if m.startswith("granite") else "1e-3"]); ax.set_xlim(4.2e-5,1.2e-3)
        ax.minorticks_off(); ax.set_title(t,fontsize=fs+2); ax.set_xlabel("Learning rate",fontsize=fs)
        i=b.index(min(b)); ax.annotate(f"best {b[i]:.3f}\n@ {x[i]:.0e}".replace("e-0","e-"),(x[i],b[i]),(0,-30),textcoords="offset points",ha="center",fontsize=fs-1,color=B,arrowprops=dict(arrowstyle="-",color=B,lw=.8))
        if m.startswith("granite"):
            ax.set_ylim(0.96,1.40); ax.annotate("7e-4: degraded from first eval\n1e-3: diverged (no loss)",(7e-4,1.3395),(2.3e-5*3,1.33),textcoords="data",fontsize=fs-1,color="#444",arrowprops=dict(arrowstyle="-",color="#888"))
        else: ax.set_ylim(1.08,1.275)
    axs[0].set_ylabel("Held-out loss",fontsize=fs)
    axs[1].text(.03,.93,"Best checkpoint (ep. 1.9)",color=B,transform=axs[1].transAxes,fontsize=fs,fontweight="bold")
    axs[1].text(.03,.84,"Last eval (ep. 2.8)",color=O,transform=axs[1].transAxes,fontsize=fs,fontweight="bold")
    f.savefig(path,dpi=200); plt.close(f)
# ---- production held-out loss by size
P={"Granite-4.1-8B":{"1,000":[(0.18, 1.9892), (0.36, 1.6199), (0.53, 1.481), (0.71, 1.38), (0.89, 1.3279), (1.04, 1.2941), (1.21, 1.2719), (1.39, 1.2415), (1.57, 1.2218), (1.75, 1.1998), (1.92, 1.1832), (2.07, 1.1753), (2.25, 1.1805), (2.43, 1.1801), (2.6, 1.177), (2.78, 1.1751)],"2,000":[(.89,1.177),(1.76,1.066),(2.64,1.070)],"3,800 (full)":[(.47,1.193),(.94,1.063),(1.41,1.022),(1.88,.987),(2.36,1.008),(2.82,1.004)]},
   "Qwen3-8B":{"1,000":[(0.18, 1.8936), (0.36, 1.5986), (0.53, 1.4697), (0.71, 1.3928), (0.89, 1.3447), (1.04, 1.3102), (1.21, 1.2857), (1.39, 1.2629), (1.57, 1.2447), (1.75, 1.2284), (1.92, 1.2127), (2.07, 1.2048), (2.25, 1.2023), (2.43, 1.1973), (2.6, 1.1938), (2.78, 1.1927)],"2,000":[(.89,1.213),(1.76,1.111),(2.64,1.098)],"3,800 (full)":[(.47,1.230),(.94,1.103),(1.41,1.064),(1.88,1.031),(2.36,1.032),(2.82,1.028)]}}
def fig_prod(path,size,fs):
    f,axs=plt.subplots(1,2,figsize=size,constrained_layout=True,sharey=True)
    for ax,(m,d) in zip(axs,P.items()):
        style(ax,fs)
        for (k,pts),c in zip(d.items(),(B,O,G)):
            x,y=zip(*pts); ax.plot(x,y,"-o",color=c,lw=2.4,ms=6 if len(pts)>6 else 7)
            ax.text(x[-1]+.06,y[-1],k,color=c,fontsize=fs-1,va="center",fontweight="bold")
        ax.set_title(m,fontsize=fs+2); ax.set_xlabel("Epoch",fontsize=fs); ax.set_xlim(0.3,3.6); ax.set_ylim(0.96,1.32)
    axs[0].set_ylabel("Held-out loss",fontsize=fs)
    f.savefig(path,dpi=200,bbox_inches="tight"); plt.close(f)
fig_lr("figs/fig_lr_sweep.png",(7.2,3.0),9); fig_lr("figs/fig_lr_sweep_slide.png",(13.3,6.2),16)
fig_prod("figs/fig_production_curves.png",(7.2,3.0),9); fig_prod("figs/fig_production_curves_slide.png",(13.3,6.2),16)
