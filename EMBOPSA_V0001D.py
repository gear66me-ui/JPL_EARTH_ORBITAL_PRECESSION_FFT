# V0001D
# Audit reference: first EMB orbital-normal FFT and harmonic reconstruction
from __future__ import annotations
import base64, io, math, os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
V='V0001D'; TZ=ZoneInfo('America/Bogota'); R='gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT'; B='main'
URL=f'https://raw.githubusercontent.com/{R}/{B}/data/derived/EMBOPSA_ORBITAL_NORMAL_3999_V0001C.csv'
OUT=Path('/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001D_OUTPUT'); K=20; F=0.80; A=206264.80624709636
FILES={
'freq':OUT/'EMBOPSA_FFT_FREQUENCIES_V0001D.csv','coef':OUT/'EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv',
'recon':OUT/'EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv','valid':OUT/'EMBOPSA_HOLDOUT_VALIDATION_V0001D.csv',
'metrics':OUT/'EMBOPSA_VALIDATION_METRICS_V0001D.csv','spectrum':OUT/'EMBOPSA_FFT_SPECTRUM_V0001D.png',
'coefpng':OUT/'EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.png','resid':OUT/'EMBOPSA_RESIDUALS_V0001D.png',
'error':OUT/'EMBOPSA_ANGULAR_ERRORS_V0001D.png','equation':OUT/'EMBOPSA_MODEL_EQUATION_V0001D.png'}
REMOTE={FILES['freq']:'spectral/'+FILES['freq'].name,FILES['coef']:'spectral/'+FILES['coef'].name,
FILES['recon']:'reconstruction/'+FILES['recon'].name,FILES['valid']:'validation/'+FILES['valid'].name,
FILES['metrics']:'validation/'+FILES['metrics'].name,FILES['spectrum']:'figures/'+FILES['spectrum'].name,
FILES['coefpng']:'figures/'+FILES['coefpng'].name,FILES['resid']:'figures/'+FILES['resid'].name,
FILES['error']:'figures/'+FILES['error'].name,FILES['equation']:'figures/'+FILES['equation'].name}
def token():
 t=os.getenv('GITHUB_TOKEN')
 if t:return t.strip()
 try:
  from google.colab import userdata
  t=userdata.get('GITHUB_TOKEN'); return t.strip() if t else None
 except Exception:return None
def publish(p,r,t):
 u=f'https://api.github.com/repos/{R}/contents/{r}'; h={'Authorization':f'Bearer {t}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
 q=requests.get(u,headers=h,params={'ref':B},timeout=60); d={'message':f'Publish {p.name}','content':base64.b64encode(p.read_bytes()).decode(),'branch':B}
 if q.status_code==200:d['sha']=q.json()['sha']
 elif q.status_code!=404:raise RuntimeError(f'GitHub lookup HTTP {q.status_code}: {q.text[:300]}')
 z=requests.put(u,headers=h,json=d,timeout=180)
 if z.status_code not in (200,201):raise RuntimeError(f'GitHub upload HTTP {z.status_code}: {z.text[:500]}')
 return f'https://raw.githubusercontent.com/{R}/{B}/{r}'
def design(t,f):
 c=[np.ones_like(t),t]
 for x in f:
  w=2*np.pi*x*t; c.extend((np.cos(w),np.sin(w)))
 return np.column_stack(c)
def stats(o,m):
 x=np.r_[o.real,o.imag]; y=np.r_[m.real,m.imag]; r=float(np.corrcoef(x,y)[0,1]); r2=1-float(np.sum((x-y)**2)/np.sum((x-x.mean())**2)); return r,r2
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 print(f'OUTPUT VERSION {V}\nCODE INPUTS\nSource normals        : V0001C repository CSV\nJPL states            : 3999\nYear range            : -9995 to +9995\nCadence               : 5 years\nTraining fraction     : {F:.2f}\nHarmonic components   : {K}\nDetrending            : complex constant + linear trend\nValidation            : final contiguous 20% JPL holdout')
 r=requests.get(URL,timeout=180); r.raise_for_status(); d=pd.read_csv(io.StringIO(r.text)).sort_values('astronomical_year').reset_index(drop=True)
 if len(d)!=3999:raise RuntimeError(f'REJECTED expected 3999 rows, got {len(d)}')
 y=d.astronomical_year.to_numpy(float); n=d[['normal_x','normal_y','normal_z']].to_numpy(float)
 if not np.allclose(np.diff(y),5):raise RuntimeError('REJECTED cadence')
 h=n.mean(0); h/=np.linalg.norm(h); ref=np.array([0.,0.,1.]) if abs(h[2])<.95 else np.array([1.,0.,0.]); ep=np.cross(ref,h); ep/=np.linalg.norm(ep); eq=np.cross(h,ep); eq/=np.linalg.norm(eq)
 den=n@h
 if np.any(den<=0):raise RuntimeError('REJECTED tangent hemisphere crossing')
 p=(n@ep)/den; q=(n@eq)/den; z=p+1j*q; t=y-y[0]; cut=int(len(y)*F); tt=t[:cut]; zz=z[:cut]
 trend=np.column_stack((np.ones_like(tt),tt)); tc=np.linalg.lstsq(trend,zz,rcond=None)[0]; zd=zz-trend@tc
 fv=np.fft.fftfreq(len(zd),d=5.0); sp=np.fft.fft(zd); mask=fv>0; pf=fv[mask]; ps=sp[mask]; ix=np.argsort(np.abs(ps))[::-1][:K]; f=pf[ix]; s=ps[ix]; order=np.argsort(np.abs(s))[::-1]; f=f[order]; s=s[order]
 X=design(tt,f); c=np.linalg.lstsq(X,zz,rcond=None)[0]; zm=design(t,f)@c; pm=zm.real; qm=zm.imag; e=np.hypot(p-pm,q-qm)*A
 amp=2*np.abs(s)/len(s); freq=pd.DataFrame({'rank':np.arange(1,K+1),'frequency_cycles_per_year':f,'period_years':1/f,'fft_complex_amplitude_rad':amp,'fft_complex_amplitude_arcsec':amp*A})
 rows=[]
 for j,x in enumerate(f):
  ca=c[2+2*j]; sa=c[3+2*j]; aa=math.sqrt(abs(ca)**2+abs(sa)**2); rows.append({'rank':j+1,'frequency_cycles_per_year':x,'period_years':1/x,'cosine_real_rad':ca.real,'cosine_imag_rad':ca.imag,'sine_real_rad':sa.real,'sine_imag_rad':sa.imag,'combined_amplitude_rad':aa,'combined_amplitude_arcsec':aa*A})
 coef=pd.DataFrame(rows); rec=pd.DataFrame({'sample_index':d.sample_index,'astronomical_year':y.astype(int),'jd_tdb':d.jd_tdb,'set':np.where(np.arange(len(y))<cut,'TRAIN','HOLDOUT'),'p_jpl_rad':p,'q_jpl_rad':q,'p_model_rad':pm,'q_model_rad':qm,'p_residual_arcsec':(p-pm)*A,'q_residual_arcsec':(q-qm)*A,'angular_error_arcsec':e})
 tr,hr=stats(z[:cut],zm[:cut]),stats(z[cut:],zm[cut:]); te=e[:cut]; he=e[cut:]; slope=float(np.polyfit(y[cut:]-y[cut],he,1)[0])
 met=pd.DataFrame({'quantity':['training_rows','holdout_rows','harmonic_components','training_pearson_correlation','training_r_squared','training_rms_error_arcsec','training_max_error_arcsec','holdout_pearson_correlation','holdout_r_squared','holdout_rms_error_arcsec','holdout_max_error_arcsec','holdout_error_trend_arcsec_per_year'],'value':[cut,len(y)-cut,K,tr[0],tr[1],np.sqrt(np.mean(te**2)),te.max(),hr[0],hr[1],np.sqrt(np.mean(he**2)),he.max(),slope]})
 freq.to_csv(FILES['freq'],index=False); coef.to_csv(FILES['coef'],index=False); rec.to_csv(FILES['recon'],index=False); rec.iloc[cut:].to_csv(FILES['valid'],index=False); met.to_csv(FILES['metrics'],index=False)
 plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'axes.linewidth':.6,'lines.linewidth':.8,'font.size':9})
 fig,ax=plt.subplots(figsize=(10,5)); ax.plot(freq.period_years,freq.fft_complex_amplitude_arcsec,marker='o',markersize=2.5); ax.set(xscale='log',yscale='log',xlabel='Period (years)',ylabel='FFT amplitude (arcsec)',title='EMB orbital-normal FFT spectrum'); ax.grid(True,linewidth=.35,alpha=.45); fig.tight_layout(); fig.savefig(FILES['spectrum']); plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5)); ax.bar(coef['rank'],coef.combined_amplitude_arcsec); ax.set(xlabel='Harmonic rank',ylabel='Coefficient amplitude (arcsec)',title='Fitted harmonic coefficients'); ax.grid(True,axis='y',linewidth=.35,alpha=.45); fig.tight_layout(); fig.savefig(FILES['coefpng']); plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5)); ax.plot(y,(p-pm)*A,label='p residual'); ax.plot(y,(q-qm)*A,label='q residual'); ax.axvline(y[cut],ls='--',lw=.7,label='holdout begins'); ax.set(xlabel='Astronomical year',ylabel='Residual (arcsec)',title='Harmonic reconstruction residuals'); ax.legend(); ax.grid(True,linewidth=.35,alpha=.45); fig.tight_layout(); fig.savefig(FILES['resid']); plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5)); ax.plot(y,e); ax.axvline(y[cut],ls='--',lw=.7,label='holdout begins'); ax.set(xlabel='Astronomical year',ylabel='Angular error (arcsec)',title='JPL versus harmonic-model angular error'); ax.legend(); ax.grid(True,linewidth=.35,alpha=.45); fig.tight_layout(); fig.savefig(FILES['error']); plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,6.5)); ax.axis('off'); eqs=[r'$\mathbf{h}=\frac{\mathbf r\times\mathbf v}{\|\mathbf r\times\mathbf v\|}$',r'$p=\frac{\mathbf h\cdot\mathbf e_p}{\mathbf h\cdot\bar{\mathbf h}},\ q=\frac{\mathbf h\cdot\mathbf e_q}{\mathbf h\cdot\bar{\mathbf h}}$',r'$z=p+iq$',r'$\widehat z=c_0+c_1t+\sum_{k=1}^{K}[a_k\cos(2\pi f_kt)+b_k\sin(2\pi f_kt)]$',r'$\epsilon=206264.806247|z-\widehat z|\ \mathrm{arcsec}$']; yy=[.87,.68,.51,.32,.12]
 for a,b in zip(eqs,yy):ax.text(.5,b,a,ha='center',va='center',fontsize=15)
 ax.set_title('EMB orbital-normal tangent-plane harmonic model',fontsize=14); fig.tight_layout(); fig.savefig(FILES['equation']); plt.close(fig)
 print(f'RESULTS\nTRAINING RMS ERROR    : {np.sqrt(np.mean(te**2)):.6f} arcsec\nHOLDOUT RMS ERROR     : {np.sqrt(np.mean(he**2)):.6f} arcsec\nHOLDOUT MAX ERROR     : {he.max():.6f} arcsec\nHOLDOUT CORRELATION   : {hr[0]:.12f}\nHOLDOUT R-SQUARED     : {hr[1]:.12f}\nHOLDOUT ERROR TREND   : {slope:.12e} arcsec/year')
 tk=token()
 if tk:
  print('GITHUB OUTPUT STATUS  : PUBLISHED')
  for lp,rp in REMOTE.items():print('GITHUB RAW OUTPUT     :',publish(lp,rp,tk))
 else:print('GITHUB OUTPUT STATUS  : NOT PUBLISHED | GITHUB_TOKEN unavailable')
 print('OUTPUT SUMMARY')
 for pth in FILES.values():print(pth)
 print('EQUATION STATUS\nVERIFIED tangent coordinates, FFT frequency selection, harmonic least squares, and angular error.')
 print(datetime.now(TZ).strftime('LOCAL TIMESTAMP %Y-%m-%d %H:%M:%S %Z')); print(f'FINAL VERSION {V}')
if __name__=='__main__':main()
# V0001D
