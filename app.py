# ap.py
 ============================================================
# ماسح موجات وولف النشطة — السوق السعودي (تداول)
# إطارات زمنية متعددة | قاعدة P5 < P3 الثابتة
# نسخة Streamlit
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st

# ──────────────────────────────────────────
# ARABIC NAMES
# ──────────────────────────────────────────

TICKER_AR = {
    "1010.SR":"بنك الرياض","1020.SR":"بنك الجزيرة","1030.SR":"الاستثمار",
    "1050.SR":"السعودي الفرنسي","1060.SR":"الأول","1080.SR":"العربي",
    "1111.SR":"مجموعة تداول","1120.SR":"الراجحي","1140.SR":"البلاد",
    "1150.SR":"الإنماء","1180.SR":"الأهلي","1182.SR":"أملاك",
    "1183.SR":"سهل","1201.SR":"تكوين","1202.SR":"صدر",
    "1210.SR":"بي سي آي","1211.SR":"معادن","1212.SR":"أسترا الصناعية",
    "1213.SR":"نقي","1214.SR":"شاكر","1301.SR":"أسلاك",
    "1302.SR":"بوان","1303.SR":"الصناعات الكهربائية","1304.SR":"الكابلات",
    "1320.SR":"أنابيب السعودية","1321.SR":"أنابيب الشرق","1322.SR":"أماك",
    "1810.SR":"سيرا","1820.SR":"مجموعة الحكير","1830.SR":"لجام للرياضة",
    "1831.SR":"لوسيد","1832.SR":"صدارة","1833.SR":"الموسى",
    "2001.SR":"كيمانول","2010.SR":"سابك","2020.SR":"سافكو",
    "2030.SR":"المصافي","2040.SR":"الخزف","2050.SR":"صافولا",
    "2060.SR":"التصنيع","2070.SR":"المراعي","2080.SR":"الغاز",
    "2081.SR":"الناقل الذكي","2082.SR":"أكوا باور","2083.SR":"مرافق",
    "2090.SR":"جبسكو","2100.SR":"وفرة","2110.SR":"الكيميائية",
    "2120.SR":"المتقدمة","2130.SR":"صدق","2140.SR":"أيان",
    "2150.SR":"زجاج","2160.SR":"أميانتيت","2170.SR":"اللجين",
    "2180.SR":"فيبكو","2190.SR":"سيسكو","2200.SR":"أنعام القابضة",
    "2210.SR":"نماء للكيماويات","2220.SR":"معدنية","2222.SR":"أرامكو",
    "2223.SR":"لوبريف","2230.SR":"الكيمائية السعودية","2240.SR":"الزامل",
    "2250.SR":"المجموعة السعودية","2270.SR":"سدافكو","2280.SR":"المراكز العربية",
    "2281.SR":"تنمية","2282.SR":"نجم","2283.SR":"المطاحن الأولى",
    "2290.SR":"ينساب","2300.SR":"صناعة الورق","2310.SR":"سبكيم",
    "2320.SR":"البابطين","2330.SR":"المتطورة","2340.SR":"العبداللطيف",
    "2350.SR":"كيان","2360.SR":"الفخارية","2370.SR":"مسك",
    "2380.SR":"بترو رابغ","2381.SR":"الحفر العربية","2382.SR":"أديس",
    "3002.SR":"جمجوم فارما","3003.SR":"أسواق المزرعة","3004.SR":"أوبال",
    "3005.SR":"ثمار","3007.SR":"زاهد","3008.SR":"الأصيل",
    "3010.SR":"أسمنت العربية","3020.SR":"أسمنت اليمامة","3030.SR":"أسمنت السعودية",
    "3040.SR":"أسمنت القصيم","3050.SR":"أسمنت الجنوبية","3060.SR":"أسمنت ينبع",
    "3080.SR":"أسمنت الشرقية","3090.SR":"أسمنت تبوك","3091.SR":"أسمنت الجوف",
    "3092.SR":"أسمنت الشمالية","4001.SR":"أسمنت نجران","4002.SR":"أسمنت الباحة",
    "4003.SR":"إكسترا","4004.SR":"دله الصحية","4005.SR":"رعاية",
    "4006.SR":"أسمنت المدينة","4007.SR":"الحمادي","4008.SR":"ساكو",
    "4009.SR":"السعودي الألماني","4011.SR":"لازوردي","4012.SR":"نسيج",
    "4013.SR":"سمنت حائل","4014.SR":"المسبوكات","4015.SR":"أسمنت أم القرى",
    "4020.SR":"العقارية","4030.SR":"البحر الأحمر","4031.SR":"لجام",
    "4040.SR":"سعودي كول","4050.SR":"ساسكو","4051.SR":"باعظيم",
    "4061.SR":"أنعام القابضة","4070.SR":"تهامة","4071.SR":"لوذان",
    "4080.SR":"سناد القابضة","4081.SR":"النايفات","4082.SR":"مرنة",
    "4090.SR":"طيبة","4100.SR":"مكة","4110.SR":"باتك",
    "4130.SR":"الباحة","4140.SR":"سبأ","4141.SR":"العمران",
    "4142.SR":"كابلات الرياض","4150.SR":"التعمير","4160.SR":"ثمار",
    "4161.SR":"بنان","4162.SR":"المنجم","4163.SR":"الدوائية",
    "4164.SR":"النهدي","4170.SR":"التطوير","4180.SR":"فتيحي",
    "4190.SR":"جرير","4191.SR":"معادنية","4192.SR":"السيف غاليري",
    "4200.SR":"الدريس","4210.SR":"نسيج","4220.SR":"إعمار",
    "4230.SR":"البوتاس","4240.SR":"فاقوس","4250.SR":"جبل عمر",
    "4260.SR":"بدجت السعودية","4261.SR":"ذيب","4262.SR":"لومي",
    "4263.SR":"سال","4270.SR":"طباعة وتغليف","4280.SR":"المملكة",
    "4290.SR":"الخليج للتدريب","4291.SR":"الوطنية للتعليم","4292.SR":"عطاء التعليمية",
    "4300.SR":"دار المعدات","4310.SR":"مدينة المعرفة","4320.SR":"الأنابيب",
    "4321.SR":"الرواد","4322.SR":"ريدان","4323.SR":"سمو",
    "4330.SR":"الرياض ريت","4331.SR":"الجزيرة ريت","4332.SR":"جدوى ريت الحرمين",
    "4333.SR":"تعليم ريت","4334.SR":"المعذر ريت","4335.SR":"مشاركة ريت",
    "4336.SR":"ملكية ريت","4337.SR":"سدكو كابيتال ريت","4338.SR":"الأهلي ريت",
    "4339.SR":"بنيان ريت","4340.SR":"الراجحي ريت","4342.SR":"جدوى ريت السعودية",
    "4344.SR":"سيكو السعودية ريت","4345.SR":"دراية ريت","4346.SR":"الإنماء ريت",
    "4347.SR":"بنان ريت","4348.SR":"الخبير ريت","4349.SR":"الصواب ريت",
    "5110.SR":"كاتريون","6001.SR":"حلواني إخوان","6002.SR":"هرفي للأغذية",
    "6004.SR":"كاد القابضة","6010.SR":"نادك","6012.SR":"ريدان الغذائية",
    "6013.SR":"التنمية الغذائية","6014.SR":"الآمار","6015.SR":"أمريكانا",
    "6020.SR":"جاكو","6040.SR":"تبوك الزراعية","6050.SR":"حائل الزراعية",
    "6060.SR":"الشرقية للتنمية","6070.SR":"الجوف الزراعية","6090.SR":"جازادكو",
    "7010.SR":"الاتصالات السعودية","7020.SR":"اتحاد الاتصالات","7030.SR":"زين السعودية",
    "7040.SR":"عذيب","7200.SR":"الحسن شاكر","7201.SR":"الصناعات المعدنية",
    "7202.SR":"اسمنت الجنوب","7203.SR":"لجين","7204.SR":"توبي",
    "8010.SR":"التعاونية","8012.SR":"جزيرة تكافل","8020.SR":"ملاذ للتأمين",
    "8030.SR":"ميدغلف للتأمين","8040.SR":"أسيج","8050.SR":"سلامة",
    "8060.SR":"ولاء","8070.SR":"الدرع العربي","8100.SR":"سايكو",
    "8120.SR":"إتحاد الخليج","8150.SR":"أسيج","8160.SR":"التأمين العربية",
    "8170.SR":"الاتحاد للتأمين","8180.SR":"الصقر للتأمين","8190.SR":"المتحدة للتأمين",
    "8200.SR":"الإعادة السعودية","8210.SR":"بوبا العربية","8230.SR":"تكافل الراجحي",
    "8240.SR":"تشب","8250.SR":"جي آي جي","8260.SR":"الخليجية العامة",
    "8270.SR":"بروج للتأمين","8280.SR":"العالمية","8300.SR":"الوطنية للتأمين",
    "8310.SR":"أمانة للتأمين","8311.SR":"عناية",
}


def get_ar(t):
    return TICKER_AR.get(t, t.replace('.SR', ''))


# ──────────────────────────────────────────
# PIVOTS
# ──────────────────────────────────────────

def find_pivots(df, order=5):
    high = df['High'].values
    low = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low, np.less_equal, order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar': int(i), 'price': high[i], 'type': 'H', 'date': df.index[i]})
    for i in sl:
        pivots.append({'bar': int(i), 'price': low[i], 'type': 'L', 'date': df.index[i]})
    pivots.sort(key=lambda x: x['bar'])
    return pivots


def get_alternating(pivots):
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p['type'] == alt[-1]['type']:
            if p['type'] == 'H' and p['price'] > alt[-1]['price']:
                alt[-1] = p
            elif p['type'] == 'L' and p['price'] < alt[-1]['price']:
                alt[-1] = p
        else:
            alt.append(p)
    return alt


# ──────────────────────────────────────────
# GEOMETRY
# ──────────────────────────────────────────

def line_at(x, x1, y1, x2, y2):
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


# ──────────────────────────────────────────
# RESAMPLE
# ──────────────────────────────────────────

def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max',
        'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()


# ──────────────────────────────────────────
# VALIDATORS
# ──────────────────────────────────────────

def validate_bull(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar'] for p in [p1, p2, p3, p4, p5]]
    if not (p1['type'] == 'L' and p2['type'] == 'H' and p3['type'] == 'L'
            and p4['type'] == 'H' and p5['type'] == 'L'):
        return None
    if v[2] >= v[0]:
        return None
    if v[3] >= v[1] or v[3] <= v[0]:
        return None
    if v[4] >= v[2]:
        return None
    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 >= 0 or s24 >= 0:
        return None
    if s13 >= s24:
        return None
    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        if (proj - v[4]) / abs(proj) < -tol:
            return None
    return {'direction': 'Bullish', 'points': [p1, p2, p3, p4, p5],
            'entry_price': v[4], 'p5_date': p5['date']}


def validate_bear(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1, p2, p3, p4, p5]]
    b = [p['bar'] for p in [p1, p2, p3, p4, p5]]
    if not (p1['type'] == 'H' and p2['type'] == 'L' and p3['type'] == 'H'
            and p4['type'] == 'L' and p5['type'] == 'H'):
        return None
    if v[2] <= v[0]:
        return None
    if v[3] <= v[1] or v[3] >= v[0]:
        return None
    if v[4] <= v[2]:
        return None
    s13 = (v[2] - v[0]) / (b[2] - b[0]) if b[2] != b[0] else 0
    s24 = (v[3] - v[1]) / (b[3] - b[1]) if b[3] != b[1] else 0
    if s13 <= 0 or s24 <= 0:
        return None
    if s13 >= s24:
        return None
    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        if (v[4] - proj) / abs(proj) < -tol:
            return None
    return {'direction': 'Bearish', 'points': [p1, p2, p3, p4, p5],
            'entry_price': v[4], 'p5_date': p5['date']}


# ──────────────────────────────────────────
# PATTERN FINDER
# ──────────────────────────────────────────

def find_active_wolfe(df, max_bars=8):
    n = len(df)
    best_bull = None
    best_bear = None
    for order in [4, 5, 6, 7]:
        piv = get_alternating(find_pivots(df, order=order))
        if len(piv) < 5:
            continue
        for offset in range(min(4, len(piv) - 4)):
            idx = len(piv) - 5 - offset
            if idx < 0:
                break
            combo = piv[idx:idx + 5]
            if n - 1 - combo[4]['bar'] > max_bars:
                continue
            r = validate_bull(*combo)
            if r and (best_bull is None or combo[4]['bar'] > best_bull['points'][4]['bar']):
                best_bull = r
            r = validate_bear(*combo)
            if r and (best_bear is None or combo[4]['bar'] > best_bear['points'][4]['bar']):
                best_bear = r
    out = []
    if best_bull:
        out.append(best_bull)
    if best_bear:
        out.append(best_bear)
    return out


# ──────────────────────────────────────────
# CHART
# ──────────────────────────────────────────

def plot_chart(ticker, df, result, tf_label):
    pts = result['points']
    direction = result['direction']
    entry = result['entry_price']
    target = result['target_price']
    is_bull = direction == 'Bullish'

    b = [p['bar'] for p in pts]
    v = [p['price'] for p in pts]
    last_bar = len(df) - 1
    last_close = df['Close'].iloc[-1]
    pct = ((target - entry) / entry) * 100

    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[4] + 30)
    df_z = df.iloc[pad_l:pad_r + 1].copy()
    off = pad_l
    zb = [x - off for x in b]
    n_z = len(df_z)

    C_W = '#0D47A1' if is_bull else '#B71C1C'
    C_T = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E = '#6A1B9A'
    C_A = '#00695C' if is_bull else '#880E4F'

    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit', wick='inherit')
    sty = mpf.make_mpf_style(marketcolors=mc, gridcolor='#EEEEEE', gridstyle='-',
                              facecolor='#FAFBFC', y_on_right=False,
                              rc={'font.size': 10, 'grid.alpha': 0.2})

    fig, axes = mpf.plot(df_z, type='candle', style=sty,
                         figsize=(16, 8), returnfig=True, volume=False)
    ax = axes[0]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W, linewidths=2.5, zorder=7)

    ext = zb[4] + 8
    ax.plot([zb[0], ext], [v[0], line_at(ext + off, b[0], v[0], b[2], v[2])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)
    ax.plot([zb[1], ext], [v[1], line_at(ext + off, b[1], v[1], b[3], v[3])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)

    fx = np.arange(zb[0], zb[4] + 1)
    f1 = [line_at(x + off, b[0], v[0], b[2], v[2]) for x in fx]
    f2 = [line_at(x + off, b[1], v[1], b[3], v[3]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    tgt_end_zb = n_z + 5
    tgt_end_bar = tgt_end_zb + off
    ax.plot([zb[0], tgt_end_zb],
            [v[0], line_at(tgt_end_bar, b[0], v[0], b[3], v[3])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)

    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)
    ax.axhline(y=entry, color=C_E, lw=0.6, ls=':', alpha=0.25)

    arrow_zb = zb[4] + max(4, (z_last - zb[4]) // 2)
    arrow_zb = min(arrow_zb, n_z + 3)
    arrow_price = line_at(arrow_zb + off, b[0], v[0], b[3], v[3])

    ax.annotate('', xy=(arrow_zb, arrow_price), xytext=(zb[4], entry),
                arrowprops=dict(arrowstyle='-|>', color=C_A, lw=3.0, mutation_scale=22,
                                connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15'),
                zorder=8)

    price_range = max(v) - min(v)
    lo = price_range * 0.08
    pct_y = arrow_price + lo if is_bull else arrow_price - lo

    ax.text(arrow_zb, pct_y, f'{pct:+.1f}%',
            fontsize=13, fontweight='bold', color=C_A,
            ha='center', va='bottom' if is_bull else 'top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_A, alpha=0.9, lw=0.8),
            zorder=10)

    lbl = ['P1', 'P2', 'P3', 'P4', 'P5']
    for i in range(5):
        is_low = pts[i]['type'] == 'L'
        dt_s = pts[i]['date'].strftime('%b %d')
        ax.annotate(f'{lbl[i]}  {v[i]:.2f}\n{dt_s}',
                    xy=(zb[i], v[i]),
                    xytext=(0, -28 if is_low else 28),
                    textcoords='offset points',
                    ha='center', va='top' if is_low else 'bottom',
                    fontsize=8.5, fontweight='bold', color=C_W,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_W, alpha=0.9, lw=0.6),
                    arrowprops=dict(arrowstyle='-', color=C_W, lw=0.6))

    dir_ar = 'صعودية' if is_bull else 'هابطة'
    ar_name = get_ar(ticker)
    emoji = '📈' if is_bull else '📉'

    ax.set_title(
        f'{emoji}   {ticker} - {ar_name}   —   موجة وولفي {dir_ar}   |   {tf_label}',
        fontsize=16, fontweight='bold', pad=16, color='#212121')
    ax.set_ylabel('')

    bc = '#E8F5E9' if is_bull else '#FFEBEE'
    bt = '#2E7D32' if is_bull else '#C62828'

    info = (
        f"  موجة وولفي {dir_ar}\n"
        f"  ─────────────────────\n"
        f"  السهم: {ar_name}\n"
        f"  الإغلاق الأخير :  {last_close:.2f}\n"
        f"  نقطة الدخول (P5) :  {entry:.2f}\n"
        f"  الهدف (1-4) :  {target:.2f}\n"
        f"  العائد المحتمل :  {pct:+.1f}%\n"
        f"  الإطار الزمني :  {tf_label}"
    )

    ax.text(0.01, 0.03, info, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', fontweight='bold', color=bt,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bc, edgecolor=bt, alpha=0.92, lw=1.2),
            zorder=10)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────
# PROCESS ONE TICKER
# ──────────────────────────────────────────

def process_one(ticker, period, interval, resample_rule):
    result_empty = (ticker, [], None)
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        return result_empty
    if df is None or len(df) < 30:
        return result_empty
    if resample_rule:
        df = resample_ohlc(df, resample_rule)
        if len(df) < 30:
            return result_empty
    found = find_active_wolfe(df)
    last_bar = len(df) - 1
    for r in found:
        b1 = r['points'][0]['bar']
        v1 = r['points'][0]['price']
        b4 = r['points'][3]['bar']
        v4 = r['points'][3]['price']
        r['target_price'] = round(line_at(last_bar, b1, v1, b4, v4), 2)
        r['last_close'] = round(df['Close'].iloc[-1], 2)
    return ticker, found, df


# ──────────────────────────────────────────
# SCAN ALL
# ──────────────────────────────────────────

def scan_all(tickers, period, interval, resample_rule, max_w=15):
    all_res = {}
    ohlc = {}
    total = len(tickers)
    bar = st.progress(0)
    status = st.empty()
    done = 0
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futs = {pool.submit(process_one, t, period, interval, resample_rule): t for t in tickers}
        for f in as_completed(futs):
            done += 1
            bar.progress(done / total)
            status.text(f"جاري مسح {done}/{total} من الرموز...")
            tk, found, df = f.result()
            if found:
                all_res[tk] = found
                ohlc[tk] = df
    status.text("✅ اكتمل المسح")
    return all_res, ohlc


# ──────────────────────────────────────────
# TICKERS
# ──────────────────────────────────────────

TICKERS = [
    '1010.SR','1020.SR','1030.SR','1050.SR','1060.SR','1080.SR','1111.SR','1120.SR',
    '1140.SR','1150.SR','1180.SR','1182.SR','1183.SR','1201.SR','1202.SR','1210.SR',
    '1211.SR','1212.SR','1213.SR','1214.SR','1301.SR','1302.SR','1303.SR','1304.SR',
    '1320.SR','1321.SR','1322.SR','1810.SR','1820.SR','1830.SR','1831.SR','1832.SR',
    '1833.SR','2001.SR','2010.SR','2020.SR','2030.SR','2040.SR','2050.SR','2060.SR',
    '2070.SR','2080.SR','2081.SR','2082.SR','2083.SR','2090.SR','2100.SR','2110.SR',
    '2120.SR','2130.SR','2140.SR','2150.SR','2160.SR','2170.SR','2180.SR','2190.SR',
    '2200.SR','2210.SR','2220.SR','2222.SR','2223.SR','2230.SR','2240.SR','2250.SR',
    '2270.SR','2280.SR','2281.SR','2282.SR','2283.SR','2290.SR','2300.SR','2310.SR',
    '2320.SR','2330.SR','2340.SR','2350.SR','2360.SR','2370.SR','2380.SR','2381.SR',
    '2382.SR','3002.SR','3003.SR','3004.SR','3005.SR','3007.SR','3008.SR',
    '3010.SR','3020.SR','3030.SR','3040.SR','3050.SR','3060.SR','3080.SR','3090.SR',
    '3091.SR','3092.SR','4001.SR','4002.SR','4003.SR','4004.SR','4005.SR','4006.SR',
    '4007.SR','4008.SR','4009.SR','4011.SR','4012.SR','4013.SR','4014.SR','4015.SR',
    '4020.SR','4030.SR','4031.SR','4040.SR','4050.SR','4051.SR','4061.SR','4070.SR',
    '4071.SR','4080.SR','4081.SR','4082.SR','4090.SR','4100.SR','4110.SR','4130.SR',
    '4140.SR','4141.SR','4142.SR','4150.SR','4160.SR','4161.SR','4162.SR','4163.SR',
    '4164.SR','4170.SR','4180.SR','4190.SR','4191.SR','4192.SR','4200.SR','4210.SR',
    '4220.SR','4230.SR','4240.SR','4250.SR','4260.SR','4261.SR','4262.SR','4263.SR',
    '4270.SR','4280.SR','4290.SR','4291.SR','4292.SR','4300.SR','4310.SR','4320.SR',
    '4321.SR','4322.SR','4323.SR','4330.SR','4331.SR','4332.SR','4333.SR','4334.SR',
    '4335.SR','4336.SR','4337.SR','4338.SR','4339.SR','4340.SR','4342.SR','4344.SR',
    '4345.SR','4346.SR','4347.SR','4348.SR','4349.SR','5110.SR','6001.SR','6002.SR',
    '6004.SR','6010.SR','6012.SR','6013.SR','6014.SR','6015.SR','6020.SR','6040.SR',
    '6050.SR','6060.SR','6070.SR','6090.SR','7010.SR','7020.SR','7030.SR','7040.SR',
    '7200.SR','7201.SR','7202.SR','7203.SR','7204.SR','8010.SR','8012.SR','8020.SR',
    '8030.SR','8040.SR','8050.SR','8060.SR','8070.SR','8100.SR','8120.SR','8150.SR',
    '8160.SR','8170.SR','8180.SR','8190.SR','8200.SR','8210.SR','8230.SR','8240.SR',
    '8250.SR','8260.SR','8270.SR','8280.SR','8300.SR','8310.SR','8311.SR',
]


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def build_df(items):
    if not items:
        return pd.DataFrame()
    rows = [{k: val for k, val in d.items() if k != '_r'} for d in items]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

TF = {
    '30m': ('30 دقيقة', '30m', '60d', None),
    '1h':  ('ساعة', '60m', '60d', None),
    '2h':  ('ساعتان', '60m', '60d', '2h'),
    '4h':  ('4 ساعات', '60m', '60d', '4h'),
    '1d':  ('يومي', '1d', '1y', None),
    '1w':  ('أسبوعي', '1wk', '5y', None),
}


def main():
    st.set_page_config(page_title="فاحص موجات وولفي — تداول", layout="wide")
    st.title("🎯 فاحص موجات وولفي — السوق السعودي")

    c1, c2 = st.columns([1, 3])
    with c1:
        tf_key = st.selectbox("الإطار الزمني", list(TF.keys()),
                              format_func=lambda k: TF[k][0], index=4)
        view = st.selectbox("نوع الموجة", ["صعودية", "هابطة", "الكل"], index=2)
        go = st.button("بدء المسح")
    with c2:
        st.markdown(
            "⚠️ **تنويه:** هذا فحص آلي لموجات وولفي ويف. "
            "لا يُعتبر توصية. يجب متابعة الحركة السعرية واستخدام إدارة رأس مال مناسبة."
        )

    if not go:
        st.info("اختر الإطار الزمني ثم اضغط **بدء المسح**.")
        return

    tf_label, interval, period, resample = TF[tf_key]
    st.subheader("⚙️ إعدادات المسح")
    st.write(f"- الإطار الزمني: **{tf_label}**")
    st.write(f"- الفترة: **{period}** ، الفاصل: **{interval}**")

    results, ohlc_data = scan_all(TICKERS, period, interval, resample)

    bulls = []
    bears = []
    for tk, pats in results.items():
        ar = get_ar(tk)
        for r in pats:
            pct = ((r['target_price'] - r['entry_price']) / r['entry_price']) * 100
            intra = interval not in ['1d', '1wk']
            if intra:
                d5 = r['points'][4]['date'].strftime('%Y-%m-%d %H:%M')
            else:
                d5 = r['points'][4]['date'].strftime('%Y-%m-%d')
            item = {
                'الرمز': tk,
                'الاسم': ar,
                'الإغلاق': r['last_close'],
                'دخول (P5)': round(r['entry_price'], 2),
                'الهدف': r['target_price'],
                'العائد %': round(pct, 1),
                'تاريخ P5': d5,
                '_r': r,
            }
            if r['direction'] == 'Bullish':
                bulls.append(item)
            else:
                bears.append(item)

    bulls.sort(key=lambda x: x['العائد %'], reverse=True)
    bears.sort(key=lambda x: x['العائد %'])

    st.subheader(f"📊 ملخص — {tf_label}")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("صعودية 📈", len(bulls))
    with m2:
        st.metric("هابطة 📉", len(bears))

    if view in ["صعودية", "الكل"]:
        st.markdown("### 📈 النماذج الصعودية")
        if bulls:
            st.dataframe(build_df(bulls), use_container_width=True)
        else:
            st.warning("لا توجد نماذج صعودية نشطة.")

    if view in ["هابطة", "الكل"]:
        st.markdown("### 📉 النماذج الهابطة")
        if bears:
            st.dataframe(build_df(bears), use_container_width=True)
        else:
            st.warning("لا توجد نماذج هابطة نشطة.")

    st.markdown("### الرسوم البيانية")

    if view in ["صعودية", "الكل"] and bulls:
        with st.expander("عرض الرسوم الصعودية"):
            for item in bulls:
                tk = item['الرمز']
                st.markdown(f"#### {tk} - {item['الاسم']} — صعودية")
                fig = plot_chart(tk, ohlc_data[tk], item['_r'], tf_label)
                st.pyplot(fig)

    if view in ["هابطة", "الكل"] and bears:
        with st.expander("عرض الرسوم الهابطة"):
            for item in bears:
                tk = item['الرمز']
                st.markdown(f"#### {tk} - {item['الاسم']} — هابطة")
                fig = plot_chart(tk, ohlc_data[tk], item['_r'], tf_label)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
