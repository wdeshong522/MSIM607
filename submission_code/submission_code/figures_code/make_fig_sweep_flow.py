import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, textwrap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
B="#2a78d6"; INK="#222222"; MUTE="#555555"; LINE="#9a9a9a"; CARD="#f5f5f5"
W,H=13.33,6.9
f=plt.figure(figsize=(W,H)); ax=f.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
def rb(x,y,w,h,fc,ec,lw=1.2,r=0.12,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls))
ax.text(W/2,H-0.38,"Staged sweep: one variable at a time, best value carried forward",ha="center",va="center",fontsize=22,color=INK,fontweight="bold")
stages=[("1  Rank",["8","16","32"],"16",1,"Flat: within 0.006 in loss. Rank 64 ran on Granite only (Qwen out of memory)."),
 ("2  Learning rate",["5e-5","1e-4","2e-4","3e-4","5e-4","7e-4","1e-3"],"3e-4",2,"3e-4 beats 2e-4 by 0.006 (Granite) and 0.018 (Qwen), at the best checkpoint only. Granite unstable at 7e-4, diverged at 1e-3."),
 ("3  Target modules",["attention only","attention + MLP"],"attention + MLP",1,"Adding the MLP lowers loss by about 0.07 (best checkpoint)."),
 ("4  Alpha",["16","32","64"],"32",1,"Small differences (at most 0.03); kept 32."),
 ("5  Epochs",["2","3","4"],"3",1,"All arms scored at about epoch 1.9, so this is not a test of training length.")]
n=5; gap=0.42; cw=(W-0.6-gap*(n-1))/n; x0=0.3; top=H-1.2; ch=4.45; cy=top-ch
for k,(name,vals,chosen,cols,note) in enumerate(stages):
    x=x0+k*(cw+gap); rb(x,cy,cw,ch,CARD,LINE)
    ax.text(x+cw/2,top-0.33,name,ha="center",va="center",fontsize=17,color=INK,fontweight="bold")
    ph=0.38; pg=0.10; ytop=top-0.75
    if cols==2: pw=(cw-0.45)/2
    else: pw=cw-0.5
    for i,v in enumerate(vals):
        r,c=(i//2,i%2) if cols==2 else (i,0)
        px=x+0.25+c*(pw+0.0 if cols==1 else pw+0.0)
        if cols==2 and len(vals)%2==1 and i==len(vals)-1: px=x+(cw-pw)/2
        py=ytop-(r+1)*ph-r*pg
        sel=(v==chosen)
        rb(px,py,pw,ph,B if sel else "white",B if sel else LINE,lw=1.4,r=0.19)
        ax.text(px+pw/2,py+ph/2,v,ha="center",va="center",fontsize=15,color="white" if sel else INK,fontweight="bold" if sel else "normal")
    ny=cy+0.12
    ax.text(x+cw/2,ny+0.02,"\n".join(textwrap.wrap(note,25)),ha="center",va="bottom",fontsize=12.5,color=MUTE,linespacing=1.25)
    if k<n-1:
        ax.add_patch(FancyArrowPatch((x+cw+0.03,cy+ch/2),(x+cw+gap-0.03,cy+ch/2),arrowstyle="-|>",mutation_scale=26,lw=3,color="#666666"))
# legend + bottom strip
ax.text(W/2,H-0.85,"Filled blue = value carried to the next stage",ha="center",va="center",fontsize=13.5,color=B)
sy=0.22; sh=0.78
rb(0.3,sy,W-0.6,sh,"white","#666666",lw=1.2,ls=(0,(4,3)))
ax.text(W/2,sy+sh/2,"32 runs, 1 GPU each, metric = held-out loss.   Then: production adapters at 1,000 / 2,000 / 3,800 examples,\nplus two extra learning-rate arms (2e-4, 5e-5) at 3,800.",ha="center",va="center",fontsize=14,color=INK,linespacing=1.4)
f.savefig("figs/fig_sweep_flow_slide.png",dpi=200); print("saved")
