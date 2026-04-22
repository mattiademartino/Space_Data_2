"""
CanSat Mission Overview — Data Quality Assessment
==================================================
Shows raw data from all 4 probes to demonstrate which datasets
are actually usable for analysis. NO external dataset comparison.
Output: ./figures/mission_overview.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = "./Cansat data utili"
SCAMSAT_DIR = "./002"
OUTPUT_DIR  = "./figures"
DUBENDORF   = 448
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {'VAMOS':'#1f77b4','SCAMSAT':'#2ca02c','GRASP':'#d62728','OBAMA':'#9467bd'}

def clean(df):
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna().reset_index(drop=True)

def baro(p, p0): return 44330*(1-(p/p0)**(1/5.255))

def find_drop(t, p):
    p_sm = pd.Series(p).rolling(5,center=True,min_periods=1).median().values
    dpdt = np.gradient(p_sm)/np.gradient(t)
    idx  = np.where(dpdt>0.1)[0]
    if not len(idx): return None, None
    gaps = np.where(np.diff(idx)>15)[0]
    S = np.concatenate([[idx[0]],idx[gaps+1]]) if len(gaps) else np.array([idx[0]])
    E = np.concatenate([idx[gaps],[idx[-1]]]) if len(gaps) else np.array([idx[-1]])
    ib = np.argmax([p[e]-p[s] for s,e in zip(S,E)])
    ds,de = S[ib],E[ib]
    apex = max(0,ds-60)+int(np.argmin(p[max(0,ds-60):ds+1]))
    post = np.where(p[de:min(len(p),de+60)]>p[apex]+30)[0]
    return t[apex], t[de+(post[0] if len(post) else 0)]

print("Loading data...")

# VAMOS
sv    = clean(pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv"))
t_v   = sv["timestamp_ms"].values/1000
p_v   = sv["pressure_hPa"].values
T_v   = sv["temperature_C"].values
co2_v = sv["co2_ppm"].values
p0_v  = (np.median(p_v[:100])+np.median(p_v[-500:]))/2
h_v   = baro(p_v, p0_v)
wv    = clean(pd.read_csv(f"{DATA_DIR}/wind_VAMOS.csv"))
tw    = wv["timestamp_ms"].values/1000
resets = np.where(np.diff(tw)<0)[0]
if len(resets): wv=wv.iloc[resets[-1]+1:].reset_index(drop=True); tw=wv["timestamp_ms"].values/1000
ws_v  = wv["wind_speed"].values
ta_v, tl_v = find_drop(t_v, p_v)
dm_v  = (t_v>=ta_v)&(t_v<=tl_v)
dwm_v = (tw>=ta_v)&(tw<=tl_v)
print(f"  VAMOS:   {len(sv)} samples, drop {dm_v.sum()} pts, h={h_v[dm_v].max():.0f} m AGL")

# GRASP
sg = clean(pd.read_csv(f"{DATA_DIR}/science_GRASP.csv", on_bad_lines='skip'))
sg.columns = [c.strip() for c in sg.columns]
t_g   = sg["timestamp [ms]"].values/1000
p_g   = sg["pressure [Pa]"].values/100
T_g   = sg["temperature [C]"].values
h_g_col = " altitude [m] " if " altitude [m] " in sg.columns else "altitude [m]"
h_g   = sg[h_g_col].values - DUBENDORF
pm25_g = sg["pm2_5 [microg/m3]"].values
pm10_g = sg["pm10_0 [microg/m3]"].values
t_g_rel = t_g - t_g[0]
print(f"  GRASP:   {len(sg)} samples, Δp={p_g.max()-p_g.min():.1f} hPa, T range={T_g.max()-T_g.min():.2f}°C")

# OBAMA
ob = pd.read_excel(f"{DATA_DIR}/OBAMA_data_decoded.xlsx", sheet_name='decoded')
for c in ['Time_s','first_pstat_avg_hPa','first_temp_avg_C','first_hum_avg_pct',
          'second_pstat_avg_hPa','second_temp_avg_C']:
    ob[c] = pd.to_numeric(ob[c], errors='coerce')
ob = ob[(ob['Time_s']>0)&(ob['Time_s']<600)&
        (ob['first_temp_avg_C'].between(-50,60))].dropna(
        subset=['Time_s','first_pstat_avg_hPa']).sort_values('Time_s').reset_index(drop=True)
t_o   = ob['Time_s'].values
p_o   = ob['first_pstat_avg_hPa'].values
T_o   = ob['first_temp_avg_C'].values
p_o2  = ob['second_pstat_avg_hPa'].values
T_o2  = ob['second_temp_avg_C'].values
p0_o  = (np.median(p_o[:3])+np.median(p_o[-3:]))/2
h_o   = baro(p_o, p0_o)
print(f"  OBAMA:   {len(ob)} samples, h_range={h_o.max()-h_o.min():.0f} m")

# SCAMSAT
SCAM_DUR=3550; SCAM_T0=828; SCAM_T1=996
scamsat_ok=False
def load_sc(fname, dtype, scale=1.0):
    path=os.path.join(SCAMSAT_DIR,fname)
    return np.fromfile(path,dtype=dtype).astype(float)*scale if os.path.exists(path) else None

alt_sc=load_sc('alt.txt','uint16',0.1); press_sc=load_sc('press.txt','float32',1.0)
temp_sc=load_sc('temp.txt','float32',1.0); pm25_sc=load_sc('pm2_5.txt','uint16',0.1)
pm10_sc=load_sc('pm10.txt','uint16',0.1)

if alt_sc is not None and press_sc is not None:
    scamsat_ok=True
    N_sc=len(alt_sc); t_sc=np.linspace(0,SCAM_DUR,N_sc)
    dm_sc=(t_sc>=SCAM_T0)&(t_sc<=SCAM_T1)
    h_sc=alt_sc-DUBENDORF; p_sc=press_sc/100
    p0_sc=np.median(p_sc[dm_sc][-20:])
    print(f"  SCAMSAT: {N_sc} samples, drop {dm_sc.sum()} pts, h={h_sc[dm_sc].max():.0f} m AGL")
else:
    print("  SCAMSAT: files not found in ./002/")

# ============================================================================
# FIGURE
# ============================================================================

print("\nGenerating figure...")
fig = plt.figure(figsize=(22, 24))
gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.52, wspace=0.3,
                        top=0.96, bottom=0.07)

def badge(ax, text, color, loc='tl'):
    x,y,ha,va = (0.03,0.97,'left','top') if loc=='tl' else (0.97,0.97,'right','top')
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va=va, ha=ha, color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, lw=0))

def infobox(ax, text, loc='tr'):
    x,ha = (0.97,'right') if 'r' in loc else (0.03,'left')
    y,va = (0.97,'top') if 't' in loc else (0.03,'bottom')
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, family='monospace',
            va=va, ha=ha, bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, lw=0.8))

# ─── ROW 0: Full mission pressure ────────────────────────────────────────────
fig.text(0.5, 0.969, 'ROW 1 — Full Mission: Pressure vs Time  (who captured the drop?)',
         ha='center', fontsize=11, fontstyle='italic', color='#555')

for col, name in enumerate(['VAMOS','GRASP','OBAMA','SCAMSAT']):
    ax = fig.add_subplot(gs[0,col])
    ax.set_facecolor('#f8f9fa')
    ax.set_title(name, fontsize=14, fontweight='bold', color=C[name], pad=5)
    ax.set_ylabel('Pressure [hPa]'); ax.grid(alpha=0.3)

    if name=='VAMOS':
        ax.plot(t_v/60, p_v, color=C[name], lw=1)
        ax.axvspan(ta_v/60, tl_v/60, color=C[name], alpha=0.2)
        ax.axvline(ta_v/60, color='green',  lw=1.5, ls='--', label='Apogee')
        ax.axvline(tl_v/60, color='orange', lw=1.5, ls='--', label='Landing')
        ax.legend(fontsize=8); ax.set_xlabel('Time [min]')
        infobox(ax, f'N = {len(sv):,}\nfs ≈ 1 Hz\nDrop visible ✓')
        badge(ax, 'PRIMARY', '#198754')

    elif name=='GRASP':
        ax.plot(t_g_rel, p_g, color=C[name], lw=1)
        ax.set_xlabel('Time from trigger [s]')
        infobox(ax, f'N = {len(sg):,}\nfs ≈ 38 Hz\nΔp = {p_g.max()-p_g.min():.0f} hPa only\n(expected ~70 hPa)')
        badge(ax, 'INCOMPLETE', '#dc3545')

    elif name=='OBAMA':
        ax.plot(t_o, p_o,  'o-', color=C[name], lw=1.5, ms=6, label='Sensor 1')
        ax.plot(t_o, p_o2, 's--',color=C[name], lw=1,   ms=5, alpha=0.5, label='Sensor 2')
        ax.legend(fontsize=8); ax.set_xlabel('Time [s]')
        infobox(ax, f'N = {len(ob)}\nfs ≈ 0.04 Hz\n46 hPa quant. steps\nDual sensors (SNR)')
        badge(ax, 'LIMITED', '#ffc107')

    elif name=='SCAMSAT':
        if scamsat_ok:
            ax.plot(t_sc/60, p_sc, color=C[name], lw=1)
            ax.axvspan(SCAM_T0/60, SCAM_T1/60, color=C[name], alpha=0.2)
            ax.axvline(SCAM_T0/60, color='green',  lw=1.5, ls='--')
            ax.axvline(SCAM_T1/60, color='orange', lw=1.5, ls='--')
            ax.set_xlabel('Time [min]')
            infobox(ax, f'N = {N_sc:,}\nfs ≈ 1 Hz\nDrop visible ✓')
            badge(ax, 'SECONDARY', '#198754')
        else:
            ax.text(0.5, 0.5, 'Binary files not found\nPlace in ./002/',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray', style='italic')
            ax.set_xlabel('Time [s]')
            badge(ax, 'PENDING', '#6c757d')

# ─── ROW 1: Altitude during drop ─────────────────────────────────────────────
fig.text(0.5, 0.775, 'ROW 2 — Altitude During Drop  (verifies actual free fall)',
         ha='center', fontsize=11, fontstyle='italic', color='#555')

for col, name in enumerate(['VAMOS','GRASP','OBAMA','SCAMSAT']):
    ax = fig.add_subplot(gs[1,col])
    ax.set_facecolor('#f8f9fa')
    ax.set_title(f'{name} — Altitude AGL', fontsize=11, fontweight='bold', color=C[name])
    ax.set_ylabel('Altitude [m AGL]'); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    if name=='VAMOS':
        tz = t_v[dm_v]-ta_v
        ax.plot(tz, h_v[dm_v], color=C[name], lw=2)
        ax.fill_between(tz, 0, h_v[dm_v], alpha=0.15, color=C[name])
        ax.axhline(0, color='saddlebrown', ls='--', lw=1, alpha=0.5)
        dur = tl_v-ta_v; rate = (h_v[dm_v].max()-h_v[dm_v].min())/dur
        infobox(ax, f'Peak:   {h_v[dm_v].max():.0f} m AGL\nDur.:   {dur:.0f} s\nRate:   {rate:.2f} m/s')
        ax.set_xlabel('Time from drop start [s]')

    elif name=='GRASP':
        ax.plot(t_g_rel, h_g, color=C[name], lw=2)
        ax.fill_between(t_g_rel, h_g.min(), h_g, alpha=0.15, color=C[name])
        ax.text(0.5, 0.55, f'Δh = {h_g.max()-h_g.min():.0f} m\n(expected ~665 m)\n\n→ NOT A REAL DROP',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))
        ax.set_xlabel('Time from trigger [s]')

    elif name=='OBAMA':
        ax.plot(t_o, h_o, 'o-', color=C[name], lw=1.5, ms=7)
        ax.fill_between(t_o, 0, h_o, alpha=0.15, color=C[name])
        infobox(ax, f'Max h:  {h_o.max():.0f} m AGL\nN pts:  {len(h_o)}\nΔh:     {h_o.max()-h_o.min():.0f} m\nQuant.: 46 hPa steps')
        ax.set_xlabel('Time [s]')

    elif name=='SCAMSAT':
        if scamsat_ok:
            tz = t_sc[dm_sc]-SCAM_T0
            ax.plot(tz, h_sc[dm_sc], color=C[name], lw=2)
            ax.fill_between(tz, 0, h_sc[dm_sc], alpha=0.15, color=C[name])
            ax.axhline(0, color='saddlebrown', ls='--', lw=1, alpha=0.5)
            dur_sc = SCAM_T1-SCAM_T0
            rate_sc = (h_sc[dm_sc].max()-h_sc[dm_sc].min())/dur_sc
            infobox(ax, f'Peak:   {h_sc[dm_sc].max():.0f} m AGL\nDur.:   {dur_sc} s\nRate:   {rate_sc:.2f} m/s')
            ax.set_xlabel('Time from drop start [s]')
        else:
            ax.text(0.5,0.5,'No data',ha='center',va='center',
                    transform=ax.transAxes,fontsize=12,color='gray')
            ax.set_xlabel('Time [s]')

# ─── ROW 2: Temperature during drop ──────────────────────────────────────────
fig.text(0.5, 0.581, 'ROW 3 — Temperature During Drop  (sensor behaviour)',
         ha='center', fontsize=11, fontstyle='italic', color='#555')

for col, name in enumerate(['VAMOS','GRASP','OBAMA','SCAMSAT']):
    ax = fig.add_subplot(gs[2,col])
    ax.set_facecolor('#f8f9fa')
    ax.set_title(f'{name} — Temperature', fontsize=11, fontweight='bold', color=C[name])
    ax.set_ylabel('Temperature [°C]'); ax.grid(alpha=0.3)

    if name=='VAMOS':
        tz = t_v[dm_v]-ta_v
        ax.plot(tz, T_v[dm_v], color=C[name], lw=2)
        infobox(ax, f'Mean:   {T_v[dm_v].mean():.1f}°C\nMin:    {T_v[dm_v].min():.1f}°C\nMax:    {T_v[dm_v].max():.1f}°C\nΔT:     {T_v[dm_v].max()-T_v[dm_v].min():.1f}°C')
        ax.set_xlabel('Time from drop start [s]')

    elif name=='GRASP':
        ax.plot(t_g_rel, T_g, color=C[name], lw=2)
        dt_g = T_g.max()-T_g.min()
        ax.text(0.5, 0.5, f'ΔT = {dt_g:.2f}°C over {t_g_rel[-1]:.0f}s\n\n→ Aircraft cabin T\n(no atmospheric signal)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))
        infobox(ax, f'Mean: {T_g.mean():.1f}°C\nΔT:   {dt_g:.2f}°C (flat!)', loc='br')
        ax.set_xlabel('Time from trigger [s]')

    elif name=='OBAMA':
        ax.plot(t_o, T_o,  'o-',  color=C[name], lw=1.5, ms=6, label='Sensor 1')
        ax.plot(t_o, T_o2, 's--', color=C[name], lw=1,   ms=5, alpha=0.6, label='Sensor 2')
        ax.legend(fontsize=9, loc='upper right')
        infobox(ax, f'S1: {T_o.min():.0f}-{T_o.max():.0f}°C\nS2: {T_o2.min():.0f}-{T_o2.max():.0f}°C\nDual sensor ✓', loc='bl')
        ax.set_xlabel('Time [s]')

    elif name=='SCAMSAT':
        if scamsat_ok and temp_sc is not None:
            tz = t_sc[dm_sc]-SCAM_T0
            ax.plot(tz, temp_sc[dm_sc], color=C[name], lw=2)
            infobox(ax, f'Mean:   {temp_sc[dm_sc].mean():.1f}°C\nMin:    {temp_sc[dm_sc].min():.1f}°C\nMax:    {temp_sc[dm_sc].max():.1f}°C\nΔT:     {temp_sc[dm_sc].max()-temp_sc[dm_sc].min():.1f}°C')
            ax.set_xlabel('Time from drop start [s]')
        else:
            ax.text(0.5,0.5,'No data',ha='center',va='center',
                    transform=ax.transAxes,fontsize=12,color='gray')
            ax.set_xlabel('Time [s]')

# ─── ROW 3: Unique parameters ─────────────────────────────────────────────────
fig.text(0.5, 0.387, 'ROW 4 — Unique Parameters per probe',
         ha='center', fontsize=11, fontstyle='italic', color='#555')

for col, name in enumerate(['VAMOS','GRASP','OBAMA','SCAMSAT']):
    ax = fig.add_subplot(gs[3,col])
    ax.set_facecolor('#f8f9fa')
    ax.grid(alpha=0.3)

    if name=='VAMOS':
        # Wind speed during drop
        if dwm_v.sum() > 5:
            tz = tw[dwm_v]-ta_v
            ax.plot(tz, ws_v[dwm_v], color=C[name], lw=2)
            ax.fill_between(tz, 0, ws_v[dwm_v], alpha=0.15, color=C[name])
            infobox(ax, f'Mean:  {ws_v[dwm_v].mean():.2f} m/s\nMax:   {ws_v[dwm_v].max():.2f} m/s\nσ:     {ws_v[dwm_v].std():.2f} m/s')
        ax.set_xlabel('Time from drop start [s]')
        ax.set_ylabel('Wind Speed [m/s]')
        ax.set_title('VAMOS — Wind Speed\n(unique parameter)', fontsize=11, fontweight='bold', color=C[name])

        # also show CO2 as inset
        ax2 = ax.inset_axes([0.0, 0.55, 0.45, 0.4])
        ax2.plot(t_v[dm_v]-ta_v, co2_v[dm_v], color='darkorange', lw=1.5)
        ax2.set_title('CO₂ [ppm]', fontsize=7)
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.2)

    elif name=='GRASP':
        ax.plot(t_g_rel, pm25_g, color=C[name], lw=2, label='PM₂.₅')
        ax.plot(t_g_rel, pm10_g, color=C[name], lw=2, ls='--', alpha=0.7, label='PM₁₀')
        ax.axhline(15, color='red', ls=':', lw=1, alpha=0.7, label='WHO PM₂.₅ (15)')
        ax.legend(fontsize=8)
        infobox(ax, f'PM₂.₅ mean: {pm25_g.mean():.1f} µg/m³\nPM₁₀ mean: {pm10_g.mean():.1f} µg/m³\n⚠ Only 176s data')
        ax.set_xlabel('Time from trigger [s]')
        ax.set_ylabel('Mass Conc. [µg/m³]')
        ax.set_title('GRASP — PM₂.₅ / PM₁₀\n(only 176s, false drop)', fontsize=11, fontweight='bold', color=C[name])

    elif name=='OBAMA':
        # Pressure difference between two sensors (SNR proxy)
        diff = np.abs(p_o - p_o2)
        ax.bar(t_o, diff, color=C[name], alpha=0.7, width=5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('|p₁ − p₂| [hPa]')
        ax.set_title('OBAMA — Dual Sensor Agreement\n(inter-sensor pressure diff)', fontsize=11, fontweight='bold', color=C[name])
        infobox(ax, f'Mean Δp: {diff.mean():.2f} hPa\nMax Δp: {diff.max():.2f} hPa\n→ Sensors agree well', loc='tr')

    elif name=='SCAMSAT':
        if scamsat_ok and pm25_sc is not None:
            tz = t_sc[dm_sc]-SCAM_T0
            ax.plot(tz, pm25_sc[dm_sc], color=C[name], lw=2, label='PM₂.₅')
            if pm10_sc is not None:
                ax.plot(tz, pm10_sc[dm_sc], color=C[name], lw=2, ls='--', alpha=0.7, label='PM₁₀')
            ax.axhline(15, color='red', ls=':', lw=1, alpha=0.7, label='WHO PM₂.₅')
            ax.legend(fontsize=8)
            infobox(ax, f'PM₂.₅ mean: {pm25_sc[dm_sc].mean():.1f} µg/m³\nPM₁₀ mean: {(pm10_sc[dm_sc].mean() if pm10_sc is not None else float("nan")):.1f} µg/m³\nFull drop ✓')
            ax.set_xlabel('Time from drop start [s]')
        else:
            ax.text(0.5,0.5,'PM data not loaded\n./002/pm2_5.txt',
                    ha='center',va='center',transform=ax.transAxes,
                    fontsize=11,color='gray',style='italic')
            ax.set_xlabel('Time [s]')
        ax.set_ylabel('Mass Conc. [µg/m³]')
        ax.set_title('SCAMSAT — PM₂.₅ / PM₁₀\n(full drop coverage)', fontsize=11, fontweight='bold', color=C[name])

# ─── ROW 4: Quality table ─────────────────────────────────────────────────────
ax_t = fig.add_subplot(gs[4,:])
ax_t.axis('off')

sc_drop  = f'{h_sc[dm_sc].max():.0f} m AGL' if scamsat_ok else 'pending'
sc_rate  = f'{(h_sc[dm_sc].max()-h_sc[dm_sc].min())/(SCAM_T1-SCAM_T0):.2f} m/s' if scamsat_ok else '—'
sc_n     = f'{N_sc:,}' if scamsat_ok else '3,550*'
sc_pm    = 'T, p, PM₂.₅, PM₁₀' if scamsat_ok else 'T, p, PM₂.₅, PM₁₀*'

rows = [
    ['VAMOS',   '1 Hz',    f'{len(sv):,}', '99 min',
     f'{h_v[dm_v].max():.0f} m AGL',
     f'{(h_v[dm_v].max()-h_v[dm_v].min())/(tl_v-ta_v):.2f} m/s',
     'T, p, CO₂, wind speed', 'PRIMARY — full drop, wind & CO₂'],
    ['SCAMSAT', '~1 Hz',   sc_n,           '~59 min',
     sc_drop, sc_rate, sc_pm, 'SECONDARY — PM data, clean drop'],
    ['GRASP',   '38 Hz',   f'{len(sg):,}', '176 s',
     f'Δp={p_g.max()-p_g.min():.0f} hPa (no drop)', '—',
     'T, p, PM₂.₅, PM₁₀', 'SPECTRAL DEMO — high fs, false trigger'],
    ['OBAMA',   '0.04 Hz', f'{len(ob)}',   '~7 min',
     f'Partial, {h_o.max():.0f} m AGL', '—',
     'T, p, humidity (×2)', 'MINIMAL — coarse quantization'],
]
cols = ['Probe','fs','N samples','Duration','Drop / Peak','Descent Rate',
        'Parameters','Role in analysis']

tbl = ax_t.table(cellText=rows, colLabels=cols, cellLoc='center',
                 loc='center', bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
tbl.auto_set_column_width(col=list(range(len(cols))))

row_fc = ['#d4edda','#d4edda','#fff3cd','#f8d7da']
for j in range(len(cols)):
    tbl[(0,j)].set_facecolor('#212529')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold', fontsize=11)
for i,rc in enumerate(row_fc):
    for j in range(len(cols)):
        tbl[(i+1,j)].set_facecolor(rc)
    tbl[(i+1,0)].set_text_props(color=list(C.values())[i], fontweight='bold', fontsize=12)
    tbl[(i+1,7)].set_text_props(fontweight='bold')
for i in range(5):
    for j in range(len(cols)):
        tbl[(i,j)].set_height(0.20)

ax_t.set_title('Data Quality Assessment — Which probes provide usable data?',
               fontsize=13, fontweight='bold', pad=10)

legend_elements = [
    mpatches.Patch(color='#d4edda', label='Usable for full analysis'),
    mpatches.Patch(color='#fff3cd', label='Usable for specific purpose only'),
    mpatches.Patch(color='#f8d7da', label='Minimal usefulness'),
]
ax_t.legend(handles=legend_elements, loc='lower center',
            bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=10, framealpha=0.9)
if not scamsat_ok:
    ax_t.text(0.5, -0.19, '* SCAMSAT data from notebook; binary files not loaded for this run.',
              ha='center', transform=ax_t.transAxes, fontsize=9, style='italic', color='gray')

fig.suptitle('Mission Overview — CanSat Drop Campaign, 6 February 2026, Dübendorf\n'
             'VAMOS (blue)  ·  GRASP (red)  ·  OBAMA (purple)  ·  SCAMSAT (green)',
             fontsize=15, fontweight='bold', y=0.978)

plt.savefig(f"{OUTPUT_DIR}/mission_overview.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {OUTPUT_DIR}/mission_overview.png")
