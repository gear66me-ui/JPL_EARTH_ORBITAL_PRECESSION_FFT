# V0001F
# PROJECT CONTRACT: NO AI-GENERATED IMAGES. PYTHON/MATPLOTLIB ONLY.
from __future__ import annotations
import io,os,base64
from pathlib import Path
import requests,numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Image
V='V0001F'; R='gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT'; B='main'; RAW=f'https://raw.githubusercontent.com/{R}/{B}'
OUT=Path('/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001F_OUTPUT'); OUT.mkdir(parents=True,exist_ok=True)
U={'f':f'{RAW}/spectral/EMBOPSA_FFT_FREQUENCIES_V0001D.csv','c':f'{RAW}/spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv','r':f'{RAW}/reconstruction/EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv'}
P={k:OUT/v for k,v in {'s':'EMBOPSA_DARK_FFT_SPECTRUM_V0001F.png','c':'EMBOPSA_DARK_COEFFICIENTS_V0001F.png','x':'EMBOPSA_DARK_COMPLEX_COEFFICIENTS_V0001F.png','r':'EMBOPSA_DARK_RESIDUALS_V0001F.png','e':'EMBOPSA_DARK_ERRORS_V0001F.png','q':'EMBOPSA_DARK_EQUATION_V0001F.png'}.items()}
C={'bg':'#05070B','panel':'#0B1220','grid':'#243247','text':'#E8EEF7','muted':'#9FB0C6','cyan':'#57D3FF','blue':'#5B8CFF','violet':'#B388FF','magenta':'#FF6EC7','orange':'#FF9F43','gold':'#FFD166','green':'#35E0A1','red':'#FF5C7A'}
def csv(u):
 r=requests.get(u,timeout=180); r.raise_for_status(); return pd.read_csv(io.StringIO(r.text))
def style(a,t,x,y):
 a.set_facecolor(C['panel']); a.set_title(t,color=C['text'],fontsize=15,fontweight='bold',pad=14); a.set_xlabel(x,color=C['muted']); a.set_ylabel(y,color=C['muted']); a.tick_params(colors=C['text']); a.grid(True,color=C['grid'],alpha=.65,lw=.6)
 [s.set_color(C['grid']) for s in a.spines.values()]
def save(fig,p):
 fig.patch.set_facecolor(C['bg']); fig.savefig(p,dpi=320,bbox_inches='tight',facecolor=C['bg']); plt.close(fig)
def main():
 plt.close('all'); plt.ioff(); plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'savefig.dpi':320,'axes.linewidth':.8,'lines.linewidth':1.25})
 f=csv(U['f']).sort_values('rank'); c=csv(U['c']).sort_values('rank'); r=csv(U['r']).sort_values('astronomical_year')
 fig,a=plt.subplots(figsize=(11.5,6.5)); a.plot(f.period_years,f.fft_complex_amplitude_arcsec,color=C['cyan'],marker='o',ms=5,mfc=C['gold'],mec=C['bg']); a.fill_between(f.period_years,f.fft_complex_amplitude_arcsec,color=C['blue'],alpha=.18); a.set_xscale('log'); a.set_yscale('log'); style(a,'EMB Orbital-Normal FFT Spectrum','Period (years, log scale)','Amplitude (arcsec, log scale)'); save(fig,P['s'])
 fig,a=plt.subplots(figsize=(11.5,6.5)); cols=[C['gold'] if x<=3 else C['blue'] if x<=10 else C['green'] for x in c['rank']]; a.bar(c['rank'],c.combined_amplitude_arcsec,color=cols,edgecolor=C['text'],lw=.45); style(a,'Fitted Harmonic Coefficients','Harmonic rank','Combined amplitude (arcsec)'); save(fig,P['c'])
 cr=(c.cosine_real_rad+1j*c.cosine_imag_rad)*206264.80624709636; sr=(c.sine_real_rad+1j*c.sine_imag_rad)*206264.80624709636
 fig,a=plt.subplots(figsize=(8.5,8.5)); a.scatter(cr.real,cr.imag,s=72,color=C['cyan'],edgecolor=C['text'],label='Cosine'); a.scatter(sr.real,sr.imag,s=72,color=C['magenta'],edgecolor=C['text'],marker='s',label='Sine'); a.axhline(0,color=C['grid'],lw=.8); a.axvline(0,color=C['grid'],lw=.8); a.set_aspect('equal',adjustable='datalim'); style(a,'Complex Harmonic Coefficients','Real component (arcsec)','Imaginary component (arcsec)'); lg=a.legend(facecolor=C['panel'],edgecolor=C['grid']); [t.set_color(C['text']) for t in lg.get_texts()]; save(fig,P['x'])
 y=r.astronomical_year.to_numpy(); cut=r.loc[r['set'].eq('HOLDOUT'),'astronomical_year'].iloc[0]
 fig,a=plt.subplots(figsize=(12.5,6.5)); a.plot(y,r.p_residual_arcsec,color=C['cyan'],label='p residual'); a.plot(y,r.q_residual_arcsec,color=C['violet'],label='q residual'); a.axvspan(cut,y.max(),color=C['gold'],alpha=.10,label='Holdout'); a.axhline(0,color=C['grid'],lw=.8); style(a,'Harmonic Reconstruction Residuals','Astronomical year','Residual (arcsec)'); lg=a.legend(facecolor=C['panel'],edgecolor=C['grid']); [t.set_color(C['text']) for t in lg.get_texts()]; save(fig,P['r'])
 tr=r['set'].eq('TRAIN'); ho=r['set'].eq('HOLDOUT'); fig,a=plt.subplots(figsize=(12.5,6.5)); a.plot(r.loc[tr,'astronomical_year'],r.loc[tr,'angular_error_arcsec'],color=C['green'],label='Training error'); a.plot(r.loc[ho,'astronomical_year'],r.loc[ho,'angular_error_arcsec'],color=C['red'],label='Holdout error'); a.axvline(cut,color=C['gold'],ls='--',lw=1.1,label='Holdout begins'); style(a,'JPL Versus 20-Harmonic Model Error','Astronomical year','Angular error (arcsec)'); lg=a.legend(facecolor=C['panel'],edgecolor=C['grid']); [t.set_color(C['text']) for t in lg.get_texts()]; save(fig,P['e'])
 fig,a=plt.subplots(figsize=(13,8)); fig.patch.set_facecolor(C['bg']); a.set_facecolor(C['panel']); a.axis('off'); a.set_xlim(0,1); a.set_ylim(0,1); a.text(.5,.88,'EMB Orbital-Normal Harmonic Reconstruction',ha='center',color=C['text'],fontsize=19,fontweight='bold')
 eq=[('h(t) = [r(t) × v(t)] / |r(t) × v(t)|',C['cyan']),('p(t) = [h(t) · eₚ] / [h(t) · h̄]',C['green']),('q(t) = [h(t) · e_q] / [h(t) · h̄]',C['green']),('z(t) = p(t) + i q(t)',C['magenta']),('ẑ(t) = c₀ + c₁t + Σₖ[aₖ cos(2πfₖt) + bₖ sin(2πfₖt)]',C['orange']),('ε(t) = 206264.806247 |z(t) − ẑ(t)| arcsec',C['red'])]
 [a.text(.5,yp,txt,ha='center',va='center',color=col,fontsize=15) for (txt,col),yp in zip(eq,[.73,.62,.52,.41,.28,.15])]; save(fig,P['q'])
 for p in P.values(): display(Image(filename=str(p)))
 plt.close('all')
if __name__=='__main__': main()
# V0001F
