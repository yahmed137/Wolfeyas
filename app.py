# app.py
# ============================================================
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
import matplotlib
import mplfinance as mpf
import streamlit as st
from bidi.algorithm import get_display
import arabic_reshaper

# ────────────────────────────────────────────────────────────
# 0. ARABIC FONT SETUP FOR MATPLOTLIB
# ────────────────────────────────────────────────────────────

def shape_arabic(text):
    """Reshape and reorder Arabic text for matplotlib rendering."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# Set a font that supports Arabic — try common ones
_ARABIC_FONTS = [
    'Tahoma', 'Arial', 'DejaVu Sans', 'Noto Sans Arabic',
    'Noto Naskh Arabic', 'Simplified Arabic', 'Traditional Arabic',
    'Amiri', 'Cairo', 'Scheherazade',
]

def _setup_arabic_font():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for fname in _ARABIC_FONTS:
        if fname in available:
            matplotlib.rcParams['font.family'] = fname
            return fname
    # fallback — use sans-serif and hope for the best
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return 'sans-serif'

_FONT_USED = _setup_arabic_font()

# ────────────────────────────────────────────────────────────
# 0b. ARABIC TICKER NAME MAP
# ────────────────────────────────────────────────────────────

TICKER_NAME_AR = {
    '1010.SR': 'الرياض',
    '1020.SR': 'الجزيرة',
    '1030.SR': 'الاستثمار',
    '1050.SR': 'البنك السعودي الفرنسي',
    '1060.SR': 'الأول',
    '1080.SR': 'العربي',
    '1111.SR': 'مجموعة تداول',
    '1120.SR': 'الراجحي',
    '1140.SR': 'البلاد',
    '1150.SR': 'الإنماء',
    '1180.SR': 'الأهلي',
    '1182.SR': 'أملاك',
    '1183.SR': 'سهل',
    '1201.SR': 'تكوين',
    '1202.SR': 'صدر',
    '1210.SR': 'بي سي آي',
    '1211.SR': 'معادن',
    '1212.SR': 'أسترا الصناعية',
    '1213.SR': 'نقي',
    '1214.SR': 'شاكر',
    '1301.SR': 'أسلاك',
    '1302.SR': 'بوان',
    '1303.SR': 'الصناعات الكهربائية',
    '1304.SR': 'الكابلات',
    '1320.SR': 'أنابيب السعودية',
    '1321.SR': 'أنابيب الشرق',
    '1322.SR': 'أماك',
    '1810.SR': 'سيرا',
    '1820.SR': 'مجموعة الحكير',
    '1830.SR': 'لجام للرياضة',
    '1831.SR': 'لوسيد (المها)',
    '1832.SR': 'صدارة',
    '1833.SR': 'الموسى',
    '2001.SR': 'كيمانول',
    '2010.SR': 'سابك',
    '2020.SR': 'سافكو',
    '2030.SR': 'المصافي',
    '2040.SR': 'الخزف',
    '2050.SR': 'صافولا',
    '2060.SR': 'التصنيع',
    '2070.SR': 'المراعي',
    '2080.SR': 'الغاز',
    '2081.SR': 'الناقل الذكي',
    '2082.SR': 'أكوا باور',
    '2083.SR': 'مرافق',
    '2090.SR': 'جبسكو',
    '2100.SR': 'وفرة',
    '2110.SR': 'الكيميائية',
    '2120.SR': 'المتقدمة',
    '2130.SR': 'صدق',
    '2140.SR': 'أيان',
    '2150.SR': 'زجاج',
    '2160.SR': 'أميانتيت',
    '2170.SR': 'اللجين',
    '2180.SR': 'فيبكو',
    '2190.SR': 'سيسكو',
    '2200.SR': 'أنعام القابضة',
    '2210.SR': 'نماء للكيماويات',
    '2220.SR': 'معدنية',
    '2222.SR': 'أرامكو',
    '2223.SR': 'لوبريف',
    '2230.SR': 'الكيمائية السعودية',
    '2240.SR': 'الزامل',
    '2250.SR': 'المجموعة السعودية',
    '2270.SR': 'سدافكو',
    '2280.SR': 'المراكز العربية',
    '2281.SR': 'تنمية',
    '2282.SR': 'نجم',
    '2283.SR': 'المطاحن الأولى',
    '2290.SR': 'ينساب',
    '2300.SR': 'صناعة الورق',
    '2310.SR': 'سبكيم',
    '2320.SR': 'البابطين',
    '2330.SR': 'المتطورة',
    '2340.SR': 'العبداللطيف',
    '2350.SR': 'كيان',
    '2360.SR': 'الفخارية',
    '2370.SR': 'مسك',
    '2380.SR': 'بترو رابغ',
    '2381.SR': 'الحفر العربية',
    '2382.SR': 'أديس',
    '3002.SR': 'جمجوم فارما',
    '3003.SR': 'أسواق المزرعة',
    '3004.SR': 'أوبال',
    '3005.SR': 'ثمار',
    '3007.SR': 'زاهد',
    '3008.SR': 'الأصيل',
    '3010.SR': 'أسمنت العربية',
    '3020.SR': 'أسمنت اليمامة',
    '3030.SR': 'أسمنت السعودية',
    '3040.SR': 'أسمنت القصيم',
    '3050.SR': 'أسمنت الجنوبية',
    '3060.SR': 'أسمنت ينبع',
    '3080.SR': 'أسمنت الشرقية',
    '3090.SR': 'أسمنت تبوك',
    '3091.SR': 'أسمنت الجوف',
    '3092.SR': 'أسمنت الشمالية',
    '4001.SR': 'أسمنت نجران',
    '4002.SR': 'أسمنت الباحة',
    '4003.SR': 'إكسترا',
    '4004.SR': 'دله الصحية',
    '4005.SR': 'رعاية',
    '4006.SR': 'أسمنت المدينة',
    '4007.SR': 'الحمادي',
    '4008.SR': 'ساكو',
    '4009.SR': 'السعودي الألماني',
    '4011.SR': 'لازوردي',
    '4012.SR': 'نسيج',
    '4013.SR': 'سمنت حائل',
    '4014.SR': 'المسبوكات',
    '4015.SR': 'أسمنت أم القرى',
    '4020.SR': 'العقارية',
    '4030.SR': 'البحر الأحمر',
    '4031.SR': 'لجام',
    '4040.SR': 'سعودي كول',
    '4050.SR': 'ساسكو',
    '4051.SR': 'باعظيم',
    '4061.SR': 'أنعام القابضة',
    '4070.SR': 'تهامة',
    '4071.SR': 'لوذان',
    '4080.SR': 'سناد القابضة',
    '4081.SR': 'النايفات',
    '4082.SR': 'مرنة',
    '4090.SR': 'طيبة',
    '4100.SR': 'مكة',
    '4110.SR': 'باتك',
    '4130.SR': 'الباحة',
    '4140.SR': 'سبأ',
    '4141.SR': 'العمران',
    '4142.SR': 'كابلات الرياض',
    '4150.SR': 'التعمير',
    '4160.SR': 'ثمار',
    '4161.SR': 'بنان',
    '4162.SR': 'المنجم',
    '4163.SR': 'الدوائية',
    '4164.SR': 'النهدي',
    '4170.SR': 'التطوير',
    '4180.SR': 'فتيحي',
    '4190.SR': 'جرير',
    '4191.SR': 'معادنية',
    '4192.SR': 'السيف غاليري',
    '4200.SR': 'الدريس',
    '4210.SR': 'نسيج',
    '4220.SR': 'إعمار',
    '4230.SR': 'البوتاس',
    '4240.SR': 'فاقوس',
    '4250.SR': 'جبل عمر',
    '4260.SR': 'بدجت السعودية',
    '4261.SR': 'ذيب',
    '4262.SR': 'لومي',
    '4263.SR': 'سال',
    '4270.SR': 'طباعة وتغليف',
    '4280.SR': 'المملكة',
    '4290.SR': 'الخليج للتدريب',
    '4291.SR': 'الوطنية للتعليم',
    '4292.SR': 'عطاء التعليمية',
    '4300.SR': 'دار المعدات',
    '4310.SR': 'مدينة المعرفة',
    '4320.SR': 'الأنابيب',
    '4321.SR': 'الرواد',
    '4322.SR': 'ريدان',
    '4323.SR': 'سمو',
    '4330.SR': 'الرياض ريت',
    '4331.SR': 'الجزيرة ريت',
    '4332.SR': 'جدوى ريت الحرمين',
    '4333.SR': 'تعليم ريت',
    '4334.SR': 'المعذر ريت',
    '4335.SR': 'مشاركة ريت',
    '4336.SR': 'ملكية ريت',
    '4337.SR': 'سدكو كابيتال ريت',
    '4338.SR': 'الأهلي ريت',
    '4339.SR': 'بنيان ريت',
    '4340.SR': 'الراجحي ريت',
    '4342.SR': 'جدوى ريت السعودية',
    '4344.SR': 'سيكو السعودية ريت',
    '4345.SR': 'دراية ريت',
    '4346.SR': 'الإنماء ريت',
    '4347.SR': 'بنان ريت',
    '4348.SR': 'الخبير ريت',
    '4349.SR': 'الصواب ريت',
    '5110.SR': 'كاتريون',
    '6001.SR': 'حلواني إخوان',
    '6002.SR': 'هرفي للأغذية',
    '6004.SR': 'كاد القابضة',
    '6010.SR': 'نادك',
    '6012.SR': 'ريدان الغذائية',
    '6013.SR': 'التنمية الغذائية',
    '6014.SR': 'الآمار',
    '6015.SR': 'أمريكانا',
    '6020.SR': 'جاكو',
    '6040.SR': 'تبوك الزراعية',
    '6050.SR': 'حائل الزراعية',
    '6060.SR': 'الشرقية للتنمية',
    '6070.SR': 'الجوف الزراعية',
    '6090.SR': 'جازادكو',
    '7010.SR': 'الاتصالات السعودية',
    '7020.SR': 'اتحاد الاتصالات',
    '7030.SR': 'زين السعودية',
    '7040.SR': 'عذيب',
    '7200.SR': 'الحسن غازي شاكر',
    '7201.SR': 'الصناعات المعدنية',
    '7202.SR': 'اسمنت الجنوب',
    '7203.SR': 'لجين',
    '7204.SR': 'توبي',
    '8010.SR': 'التعاونية',
    '8012.SR': 'جزيرة تكافل',
    '8020.SR': 'ملاذ للتأمين',
    '8030.SR': 'ميدغلف للتأمين',
    '8040.SR': 'أسيج',
    '8050.SR': 'سلامة',
    '8060.SR': 'ولاء',
    '8070.SR': 'الدرع العربي',
    '8100.SR': 'سايكو',
    '8120.SR': 'إتحاد الخليج',
    '8150.SR': 'أسيج',
    '8160.SR': 'التأمين العربية',
    '8170.SR': 'الاتحاد للتأمين',
    '8180.SR': 'الصقر للتأمين',
    '8190.SR': 'المتحدة للتأمين',
    '8200.SR': 'الإعادة السعودية',
    '8210.SR': 'بوبا العربية',
    '8230.SR': 'تكافل الراجحي',
    '8240.SR': 'تشب',
    '8250.SR': 'جي آي جي',
    '8260.SR': 'الخليجية العامة',
    '8270.SR': 'بروج للتأمين',
    '8280.SR': 'العالمية',
    '8300.SR': 'الوطنية للتأمين',
    '8310.SR': 'أمانة للتأمين',
    '8311.SR': 'عناية',
}

def get_ar_name(ticker):
    """Return Arabic company name or ticker code if not found."""
    return TICKER_NAME_AR.get(ticker, ticker.replace('.SR', ''))

# ────────────────────────────────────────────────────────────
# 1. SWING PIVOT DETECTION
# ────────────────────────────────────────────────────────────

def find_pivots(df, order=5):
    high = df['High'].values
    low  = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low,  np.less_equal,    order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar':int(i),'price':high[i],'type':'H','date':df.index[i]})
    for i in sl:
        pivots.append({'bar':int(i),'price':low[i],'type':'L','date':df.index[i]})
    pivots.sort(key=lambda x: x['bar'])
    return pivots

def get_alternating_pivots(pivots):
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p['type'] == alt[-1]['type']:
            if p['type']=='H' and p['price']>alt[-1]['price']:
                alt[-1]=p
            elif p['type']=='L' and p['price']<alt[-1]['price']:
                alt[-1]=p
        else:
            alt.append(p)
    return alt

# ────────────────────────────────────────────────────────────
# 2. GEOMETRY HELPER
# ────────────────────────────────────────────────────────────

def line_at(x, x1, y1, x2, y2):
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

# ────────────────────────────────────────────────────────────
# 3. RESAMPLE INTRADAY
# ────────────────────────────────────────────────────────────

def resample_ohlc(df, rule):
    df_r = df.resample(rule).agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }).dropna()
    return df_r

# ────────────────────────────────────────────────────────────
# 4. WOLFE WAVE VALIDATORS  (FIXED P5 RULES)
# ────────────────────────────────────────────────────────────

def validate_bullish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1,p2,p3,p4,p5]]
    b = [p['bar']   for p in [p1,p2,p3,p4,p5]]

    if not (p1['type']=='L' and p2['type']=='H' and p3['type']=='L'
            and p4['type']=='H' and p5['type']=='L'):
        return None

    if v[2] >= v[0]:
        return None

    if v[3] >= v[1] or v[3] <= v[0]:
        return None

    if v[4] >= v[2]:
        return None

    s13 = (v[2]-v[0]) / (b[2]-b[0]) if b[2]!=b[0] else 0
    s24 = (v[3]-v[1]) / (b[3]-b[1]) if b[3]!=b[1] else 0
    if s13 >= 0 or s24 >= 0:
        return None

    if s13 >= s24:
        return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        deviation = (proj - v[4]) / abs(proj)
        if deviation < -tol:
            return None

    return {'direction':'صاعد','points':[p1,p2,p3,p4,p5],
            'entry_price':v[4],'p5_date':p5['date']}

def validate_bearish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1,p2,p3,p4,p5]]
    b = [p['bar']   for p in [p1,p2,p3,p4,p5]]

    if not (p1['type']=='H' and p2['type']=='L' and p3['type']=='H'
            and p4['type']=='L' and p5['type']=='H'):
        return None

    if v[2] <= v[0]:
        return None

    if v[3] <= v[1] or v[3] >= v[0]:
        return None

    if v[4] <= v[2]:
        return None

    s13 = (v[2]-v[0]) / (b[2]-b[0]) if b[2]!=b[0] else 0
    s24 = (v[3]-v[1]) / (b[3]-b[1]) if b[3]!=b[1] else 0
    if s13 <= 0 or s24 <= 0:
        return None

    if s13 >= s24:
        return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        deviation = (v[4] - proj) / abs(proj)
        if deviation < -tol:
            return None

    return {'direction':'هابط','points':[p1,p2,p3,p4,p5],
            'entry_price':v[4],'p5_date':p5['date']}

# ────────────────────────────────────────────────────────────
# 5. ACTIVE PATTERN FINDER
# ────────────────────────────────────────────────────────────

def find_active_wolfe(df, max_bars_since_p5=8, pivot_orders=[4,5,6,7]):
    n = len(df)
    best_bull = None
    best_bear = None
    for order in pivot_orders:
        piv = get_alternating_pivots(find_pivots(df, order=order))
        if len(piv)<5:
            continue
        for offset in range(min(4, len(piv)-4)):
            idx = len(piv)-5-offset
            if idx<0:
                break
            combo = piv[idx:idx+5]
            if n-1-combo[4]['bar'] > max_bars_since_p5:
                continue
            r = validate_bullish(*combo)
            if r and (best_bull is None or combo[4]['bar']>best_bull['points'][4]['bar']):
                best_bull = r
            r = validate_bearish(*combo)
            if r and (best_bear is None or combo[4]['bar']>best_bear['points'][4]['bar']):
                best_bear = r
    out = []
    if best_bull:
        out.append(best_bull)
    if best_bear:
        out.append(best_bear)
    return out

# ────────────────────────────────────────────────────────────
# 6. CHART  (returns fig for Streamlit)
# ────────────────────────────────────────────────────────────

def plot_wolfe_chart(ticker, df, result, tf_label):
    pts       = result['points']
    direction = result['direction']
    entry     = result['entry_price']
    target    = result['target_price']
    is_bull   = direction == 'صاعد'

    ar_name   = get_ar_name(ticker)
    ar_name_shaped = shape_arabic(ar_name)
    direction_shaped = shape_arabic(direction)
    tf_shaped = shape_arabic(tf_label)

    b = [p['bar']   for p in pts]
    v = [p['price'] for p in pts]
    last_bar   = len(df) - 1
    last_close = df['Close'].iloc[-1]
    pct        = ((target - entry) / entry) * 100

    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[4] + 30)
    df_z  = df.iloc[pad_l:pad_r+1].copy()
    off   = pad_l
    zb    = [x - off for x in b]
    n_z   = len(df_z)

    C_W  = '#0D47A1' if is_bull else '#B71C1C'
    C_T  = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E  = '#6A1B9A'
    C_A  = '#00695C' if is_bull else '#880E4F'

    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350',
                                edge='inherit', wick='inherit')
    sty = mpf.make_mpf_style(
        marketcolors=mc, gridcolor='#EEEEEE', gridstyle='-',
        facecolor='#FAFBFC', y_on_right=False,
        rc={'font.size':10, 'grid.alpha':0.2}
    )

    fig, axes = mpf.plot(df_z, type='candle', style=sty,
                         figsize=(16, 8), returnfig=True, volume=False)
    ax = axes[0]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W, linewidths=2.5, zorder=7)

    ext = zb[4] + 8
    ax.plot([zb[0], ext],
            [v[0], line_at(ext+off, b[0],v[0], b[2],v[2])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)

    ax.plot([zb[1], ext],
            [v[1], line_at(ext+off, b[1],v[1], b[3],v[3])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)

    fx = np.arange(zb[0], zb[4]+1)
    f1 = [line_at(x+off, b[0],v[0], b[2],v[2]) for x in fx]
    f2 = [line_at(x+off, b[1],v[1], b[3],v[3]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    tgt_end_zb  = n_z + 5
    tgt_end_bar = tgt_end_zb + off
    ax.plot([zb[0], tgt_end_zb],
            [v[0], line_at(tgt_end_bar, b[0],v[0], b[3],v[3])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)

    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)

    ax.axhline(y=entry, color=C_E, lw=0.6, ls=':', alpha=0.25)

    arrow_land_zb = zb[4] + max(4, (z_last - zb[4]) // 2)
    arrow_land_zb = min(arrow_land_zb, n_z + 3)
    arrow_land_price = line_at(arrow_land_zb + off, b[0],v[0], b[3],v[3])

    ax.annotate(
        '', xy=(arrow_land_zb, arrow_land_price), xytext=(zb[4], entry),
        arrowprops=dict(
            arrowstyle='-|>', color=C_A, lw=3.0, mutation_scale=22,
            connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15'
        ), zorder=8)

    price_range = max(v) - min(v)
    label_offset = price_range * 0.08
    if is_bull:
        pct_y = arrow_land_price + label_offset
    else:
        pct_y = arrow_land_price - label_offset

    pct = ((target - entry) / entry) * 100
    ax.text(arrow_land_zb, pct_y, f'{pct:+.1f}%',
            fontsize=13, fontweight='bold', color=C_A,
            ha='center', va='bottom' if is_bull else 'top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C_A, alpha=0.9, lw=0.8),
            zorder=10)

    labels_txt = ['P1','P2','P3','P4','P5']
    for i in range(5):
        is_low = pts[i]['type'] == 'L'
        dt_str = pts[i]['date'].strftime('%b %d')
        ax.annotate(
            f'{labels_txt[i]}  {v[i]:.2f}\n{dt_str}',
            xy=(zb[i], v[i]),
            xytext=(0, -28 if is_low else 28),
            textcoords='offset points',
            ha='center', va='top' if is_low else 'bottom',
            fontsize=8.5, fontweight='bold', color=C_W,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C_W, alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle='-', color=C_W, lw=0.6))

    # Title with Arabic name
    emoji = '📈' if is_bull else '📉'
    title_text = (
        f'{emoji}   {ticker}  -  {ar_name_shaped}   |   '
        f'{direction_shaped}   |   {tf_shaped}'
    )
    ax.set_title(title_text,
                 fontsize=16, fontweight='bold', pad=16, color='#212121')
    ax.set_ylabel('')

    bc = '#E8F5E9' if is_bull else '#FFEBEE'
    bt = '#2E7D32' if is_bull else '#C62828'

    # Info box with Arabic labels
    lbl_direction = shape_arabic('موجة وولف ' + direction)
    lbl_name      = shape_arabic('السهم: ' + ar_name)
    lbl_close     = shape_arabic('آخر إغلاق')
    lbl_entry     = shape_arabic('دخول (P5)')
    lbl_target    = shape_arabic('هدف (1→4)')
    lbl_potential = shape_arabic('النسبة المتوقعة')
    lbl_tf        = shape_arabic('الإطار الزمني: ' + tf_label)

    info = (
        f"  {lbl_direction}\n"
        f"  ─────────────────────\n"
        f"  {lbl_name}\n"
        f"  {lbl_close} :  {last_close:.2f}\n"
        f"  {lbl_entry} :  {entry:.2f}\n"
        f"  {lbl_target} :  {target:.2f}\n"
        f"  {lbl_potential} :  {pct:+.1f}%\n"
        f"  {lbl_tf}"
    )

    ax.text(0.01, 0.03, info,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold', color=bt,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bc,
                      edgecolor=bt, alpha=0.92, lw=1.2),
            zorder=10)

    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────
# 7. TICKER PROCESSING
# ────────────────────────────────────────────────────────────

def process_ticker(ticker, period, interval, resample_rule=None):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or len(df) < 30:
            return ticker, [], None

        if resample_rule:
            df = resample_ohlc(df, resample_rule)
            if len(df) < 30:
                return ticker, [], None

        found = find_active_wolfe(df, max_bars_since_p5=8)
        last_bar = len(df) - 1
        for r in found:
            b1=r['points'][0]['bar']; v1=r['points'][0]['price']
            b4=r['points'][3]['bar']; v4=r['points'][3]['price']
            r['target_price'] = round(line_at(last_bar, b1,v1, b4,v4), 2)
            r['last_close']   = round(df[    '2130.SR': 'صدق',
    '2140.SR': 'أيان',
    '2150.SR': 'زجاج',
    '2160.SR': 'أميانتيت',
    '2170.SR': 'اللجين',
    '2180.SR': 'فيبكو',
    '2190.SR': 'سيسكو',
    '2200.SR': 'أنعام القابضة',
    '2210.SR': 'نماء للكيماويات',
    '2220.SR': 'معدنية',
    '2222.SR': 'أرامكو',
    '2223.SR': 'لوبريف',
    '2230.SR': 'الكيمائية السعودية',
    '2240.SR': 'الزامل',
    '2250.SR': 'المجموعة السعودية',
    '2270.SR': 'سدافكو',
    '2280.SR': 'المراكز العربية',
    '2281.SR': 'تنمية',
    '2282.SR': 'نجم',
    '2283.SR': 'المطاحن الأولى',
    '2290.SR': 'ينساب',
    '2300.SR': 'صناعة الورق',
    '2310.SR': 'سبكيم',
    '2320.SR': 'البابطين',
    '2330.SR': 'المتطورة',
    '2340.SR': 'العبداللطيف',
    '2350.SR': 'كيان',
    '2360.SR': 'الفخارية',
    '2370.SR': 'مسك',
    '2380.SR': 'بترو رابغ',
    '2381.SR': 'الحفر العربية',
    '2382.SR': 'أديس',
    '3002.SR': 'جمجوم فارما',
    '3003.SR': 'أسواق المزرعة',
    '3004.SR': 'أوبال',
    '3005.SR': 'ثمار',
    '3007.SR': 'زاهد',
    '3008.SR': 'الأصيل',
    '3010.SR': 'أسمنت العربية',
    '3020.SR': 'أسمنت اليمامة',
    '3030.SR': 'أسمنت السعودية',
    '3040.SR': 'أسمنت القصيم',
    '3050.SR': 'أسمنت الجنوبية',
    '3060.SR': 'أسمنت ينبع',
    '3080.SR': 'أسمنت الشرقية',
    '3090.SR': 'أسمنت تبوك',
    '3091.SR': 'أسمنت الجوف',
    '3092.SR': 'أسمنت الشمالية',
    '4001.SR': 'أسمنت نجران',
    '4002.SR': 'أسمنت الباحة',
    '4003.SR': 'إكسترا',
    '4004.SR': 'دله الصحية',
    '4005.SR': 'رعاية',
    '4006.SR': 'أسمنت المدينة',
    '4007.SR': 'الحمادي',
    '4008.SR': 'ساكو',
    '4009.SR': 'السعودي الألماني',
    '4011.SR': 'لازوردي',
    '4012.SR': 'نسيج',
    '4013.SR': 'سمنت حائل',
    '4014.SR': 'المسبوكات',
    '4015.SR': 'أسمنت أم القرى',
    '4020.SR': 'العقارية',
    '4030.SR': 'البحر الأحمر',
    '4031.SR': 'لجام',
    '4040.SR': 'سعودي كول',
    '4050.SR': 'ساسكو',
    '4051.SR': 'باعظيم',
    '4061.SR': 'أنعام القابضة',
    '4070.SR': 'تهامة',
    '4071.SR': 'لوذان',
    '4080.SR': 'سناد القابضة',
    '4081.SR': 'النايفات',
    '4082.SR': 'مرنة',
    '4090.SR': 'طيبة',
    '4100.SR': 'مكة',
    '4110.SR': 'باتك',
    '4130.SR': 'الباحة',
    '4140.SR': 'سبأ',
    '4141.SR': 'العمران',
    '4142.SR': 'كابلات الرياض',
    '4150.SR': 'التعمير',
    '4160.SR': 'ثمار',
    '4161.SR': 'بنان',
    '4162.SR': 'المنجم',
    '4163.SR': 'الدوائية',
    '4164.SR': 'النهدي',
    '4170.SR': 'التطوير',
    '4180.SR': 'فتيحي',
    '4190.SR': 'جرير',
    '4191.SR': 'معادنية',
    '4192.SR': 'السيف غاليري',
    '4200.SR': 'الدريس',
    '4210.SR': 'نسيج',
    '4220.SR': 'إعمار',
    '4230.SR': 'البوتاس',
    '4240.SR': 'فاقوس',
    '4250.SR': 'جبل عمر',
    '4260.SR': 'بدجت السعودية',
    '4261.SR': 'ذيب',
    '4262.SR': 'لومي',
    '4263.SR': 'سال',
    '4270.SR': 'طباعة وتغليف',
    '4280.SR': 'المملكة',
    '4290.SR': 'الخليج للتدريب',
    '4291.SR': 'الوطنية للتعليم',
    '4292.SR': 'عطاء التعليمية',
    '4300.SR': 'دار المعدات',
    '4310.SR': 'مدينة المعرفة',
    '4320.SR': 'الأنابيب',
    '4321.SR': 'الرواد',
    '4322.SR': 'ريدان',
    '4323.SR': 'سمو',
    '4330.SR': 'الرياض ريت',
    '4331.SR': 'الجزيرة ريت',
    '4332.SR': 'جدوى ريت الحرمين',
    '4333.SR': 'تعليم ريت',
    '4334.SR': 'المعذر ريت',
    '4335.SR': 'مشاركة ريت',
    '4336.SR': 'ملكية ريت',
    '4337.SR': 'سدكو كابيتال ريت',
    '4338.SR': 'الأهلي ريت',
    '4339.SR': 'بنيان ريت',
    '4340.SR': 'الراجحي ريت',
    '4342.SR': 'جدوى ريت السعودية',
    '4344.SR': 'سيكو السعودية ريت',
    '4345.SR': 'دراية ريت',
    '4346.SR': 'الإنماء ريت',
    '4347.SR': 'بنان ريت',
    '4348.SR': 'الخبير ريت',
    '4349.SR': 'الصواب ريت',
    '5110.SR': 'كاتريون',
    '6001.SR': 'حلواني إخوان',
    '6002.SR': 'هرفي للأغذية',
    '6004.SR': 'كاد القابضة',
    '6010.SR': 'نادك',
    '6012.SR': 'ريدان الغذائية',
    '6013.SR': 'التنمية الغذائية',
    '6014.SR': 'الآمار',
    '6015.SR': 'أمريكانا',
    '6020.SR': 'جاكو',
    '6040.SR': 'تبوك الزراعية',
    '6050.SR': 'حائل الزراعية',
    '6060.SR': 'الشرقية للتنمية',
    '6070.SR': 'الجوف الزراعية',
    '6090.SR': 'جازادكو',
    '7010.SR': 'الاتصالات السعودية',
    '7020.SR': 'اتحاد الاتصالات',
    '7030.SR': 'زين السعودية',
    '7040.SR': 'عذيب',
    '7200.SR': 'الحسن غازي شاكر',
    '7201.SR': 'الصناعات المعدنية',
    '7202.SR': 'اسمنت الجنوب',
    '7203.SR': 'لجين',
    '7204.SR': 'توبي',
    '8010.SR': 'التعاونية',
    '8012.SR': 'جزيرة تكافل',
    '8020.SR': 'ملاذ للتأمين',
    '8030.SR': 'ميدغلف للتأمين',
    '8040.SR': 'أسيج',
    '8050.SR': 'سلامة',
    '8060.SR': 'ولاء',
    '8070.SR': 'الدرع العربي',
    '8100.SR': 'سايكو',
    '8120.SR': 'إتحاد الخليج',
    '8150.SR': 'أسيج',
    '8160.SR': 'التأمين العربية',
    '8170.SR': 'الاتحاد للتأمين',
    '8180.SR': 'الصقر للتأمين',
    '8190.SR': 'المتحدة للتأمين',
    '8200.SR': 'الإعادة السعودية',
    '8210.SR': 'بوبا العربية',
    '8230.SR': 'تكافل الراجحي',
    '8240.SR': 'تشب',
    '8250.SR': 'جي آي جي',
    '8260.SR': 'الخليجية العامة',
    '8270.SR': 'بروج للتأمين',
    '8280.SR': 'العالمية',
    '8300.SR': 'الوطنية للتأمين',
    '8310.SR': 'أمانة للتأمين',
    '8311.SR': 'عناية',
}

def get_ar_name(ticker):
    """Return Arabic company name or ticker code if not found."""
    return TICKER_NAME_AR.get(ticker, ticker.replace('.SR', ''))

# ────────────────────────────────────────────────────────────
# 1. SWING PIVOT DETECTION
# ────────────────────────────────────────────────────────────

def find_pivots(df, order=5):
    high = df['High'].values
    low  = df['Low'].values
    sh = argrelextrema(high, np.greater_equal, order=order)[0]
    sl = argrelextrema(low,  np.less_equal,    order=order)[0]
    pivots = []
    for i in sh:
        pivots.append({'bar':int(i),'price':high[i],'type':'H','date':df.index[i]})
    for i in sl:
        pivots.append({'bar':int(i),'price':low[i],'type':'L','date':df.index[i]})
    pivots.sort(key=lambda x: x['bar'])
    return pivots

def get_alternating_pivots(pivots):
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p['type'] == alt[-1]['type']:
            if p['type']=='H' and p['price']>alt[-1]['price']:
                alt[-1]=p
            elif p['type']=='L' and p['price']<alt[-1]['price']:
                alt[-1]=p
        else:
            alt.append(p)
    return alt

# ────────────────────────────────────────────────────────────
# 2. GEOMETRY HELPER
# ────────────────────────────────────────────────────────────

def line_at(x, x1, y1, x2, y2):
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

# ────────────────────────────────────────────────────────────
# 3. RESAMPLE INTRADAY
# ────────────────────────────────────────────────────────────

def resample_ohlc(df, rule):
    df_r = df.resample(rule).agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }).dropna()
    return df_r

# ────────────────────────────────────────────────────────────
# 4. WOLFE WAVE VALIDATORS  (FIXED P5 RULES)
# ────────────────────────────────────────────────────────────

def validate_bullish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1,p2,p3,p4,p5]]
    b = [p['bar']   for p in [p1,p2,p3,p4,p5]]

    if not (p1['type']=='L' and p2['type']=='H' and p3['type']=='L'
            and p4['type']=='H' and p5['type']=='L'):
        return None

    if v[2] >= v[0]:
        return None

    if v[3] >= v[1] or v[3] <= v[0]:
        return None

    if v[4] >= v[2]:
        return None

    s13 = (v[2]-v[0]) / (b[2]-b[0]) if b[2]!=b[0] else 0
    s24 = (v[3]-v[1]) / (b[3]-b[1]) if b[3]!=b[1] else 0
    if s13 >= 0 or s24 >= 0:
        return None

    if s13 >= s24:
        return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        deviation = (proj - v[4]) / abs(proj)
        if deviation < -tol:
            return None

    return {'direction':'صاعد','points':[p1,p2,p3,p4,p5],
            'entry_price':v[4],'p5_date':p5['date']}

def validate_bearish(p1, p2, p3, p4, p5, tol=0.03):
    v = [p['price'] for p in [p1,p2,p3,p4,p5]]
    b = [p['bar']   for p in [p1,p2,p3,p4,p5]]

    if not (p1['type']=='H' and p2['type']=='L' and p3['type']=='H'
            and p4['type']=='L' and p5['type']=='H'):
        return None

    if v[2] <= v[0]:
        return None

    if v[3] <= v[1] or v[3] >= v[0]:
        return None

    if v[4] <= v[2]:
        return None

    s13 = (v[2]-v[0]) / (b[2]-b[0]) if b[2]!=b[0] else 0
    s24 = (v[3]-v[1]) / (b[3]-b[1]) if b[3]!=b[1] else 0
    if s13 <= 0 or s24 <= 0:
        return None

    if s13 >= s24:
        return None

    proj = line_at(b[4], b[0], v[0], b[2], v[2])
    if proj != 0:
        deviation = (v[4] - proj) / abs(proj)
        if deviation < -tol:
            return None

    return {'direction':'هابط','points':[p1,p2,p3,p4,p5],
            'entry_price':v[4],'p5_date':p5['date']}

# ────────────────────────────────────────────────────────────
# 5. ACTIVE PATTERN FINDER
# ────────────────────────────────────────────────────────────

def find_active_wolfe(df, max_bars_since_p5=8, pivot_orders=[4,5,6,7]):
    n = len(df)
    best_bull = None
    best_bear = None
    for order in pivot_orders:
        piv = get_alternating_pivots(find_pivots(df, order=order))
        if len(piv)<5:
            continue
        for offset in range(min(4, len(piv)-4)):
            idx = len(piv)-5-offset
            if idx<0:
                break
            combo = piv[idx:idx+5]
            if n-1-combo[4]['bar'] > max_bars_since_p5:
                continue
            r = validate_bullish(*combo)
            if r and (best_bull is None or combo[4]['bar']>best_bull['points'][4]['bar']):
                best_bull = r
            r = validate_bearish(*combo)
            if r and (best_bear is None or combo[4]['bar']>best_bear['points'][4]['bar']):
                best_bear = r
    out = []
    if best_bull:
        out.append(best_bull)
    if best_bear:
        out.append(best_bear)
    return out

# ────────────────────────────────────────────────────────────
# 6. CHART  (returns fig for Streamlit)
# ────────────────────────────────────────────────────────────

def plot_wolfe_chart(ticker, df, result, tf_label):
    pts       = result['points']
    direction = result['direction']
    entry     = result['entry_price']
    target    = result['target_price']
    is_bull   = direction == 'صاعد'

    ar_name   = get_ar_name(ticker)
    ar_name_shaped = shape_arabic(ar_name)
    direction_shaped = shape_arabic(direction)
    tf_shaped = shape_arabic(tf_label)

    b = [p['bar']   for p in pts]
    v = [p['price'] for p in pts]
    last_bar   = len(df) - 1
    last_close = df['Close'].iloc[-1]
    pct        = ((target - entry) / entry) * 100

    pad_l = max(0, b[0] - 10)
    pad_r = min(last_bar, b[4] + 30)
    df_z  = df.iloc[pad_l:pad_r+1].copy()
    off   = pad_l
    zb    = [x - off for x in b]
    n_z   = len(df_z)

    C_W  = '#0D47A1' if is_bull else '#B71C1C'
    C_T  = '#2E7D32' if is_bull else '#C62828'
    C_24 = '#E65100'
    C_E  = '#6A1B9A'
    C_A  = '#00695C' if is_bull else '#880E4F'

    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350',
                                edge='inherit', wick='inherit')
    sty = mpf.make_mpf_style(
        marketcolors=mc, gridcolor='#EEEEEE', gridstyle='-',
        facecolor='#FAFBFC', y_on_right=False,
        rc={'font.size':10, 'grid.alpha':0.2}
    )

    fig, axes = mpf.plot(df_z, type='candle', style=sty,
                         figsize=(16, 8), returnfig=True, volume=False)
    ax = axes[0]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

    ax.plot(zb, v, color=C_W, lw=2.5, zorder=6, alpha=0.8)
    ax.scatter(zb, v, s=120, c='white', edgecolors=C_W, linewidths=2.5, zorder=7)

    ext = zb[4] + 8
    ax.plot([zb[0], ext],
            [v[0], line_at(ext+off, b[0],v[0], b[2],v[2])],
            color=C_W, lw=1.0, ls='--', alpha=0.3)

    ax.plot([zb[1], ext],
            [v[1], line_at(ext+off, b[1],v[1], b[3],v[3])],
            color=C_24, lw=1.0, ls='--', alpha=0.3)

    fx = np.arange(zb[0], zb[4]+1)
    f1 = [line_at(x+off, b[0],v[0], b[2],v[2]) for x in fx]
    f2 = [line_at(x+off, b[1],v[1], b[3],v[3]) for x in fx]
    ax.fill_between(fx, f1, f2, alpha=0.04, color=C_W)

    tgt_end_zb  = n_z + 5
    tgt_end_bar = tgt_end_zb + off
    ax.plot([zb[0], tgt_end_zb],
            [v[0], line_at(tgt_end_bar, b[0],v[0], b[3],v[3])],
            color=C_T, lw=3.0, ls='-.', alpha=0.85, zorder=5)

    z_last = min(last_bar - off, n_z - 1)
    ax.plot(z_last, target, marker='D', ms=14, color=C_T,
            markeredgecolor='white', markeredgewidth=2, zorder=9)
    ax.axhline(y=target, color=C_T, lw=0.6, ls=':', alpha=0.25)

    ax.axhline(y=entry, color=C_E, lw=0.6, ls=':', alpha=0.25)

    arrow_land_zb = zb[4] + max(4, (z_last - zb[4]) // 2)
    arrow_land_zb = min(arrow_land_zb, n_z + 3)
    arrow_land_price = line_at(arrow_land_zb + off, b[0],v[0], b[3],v[3])

    ax.annotate(
        '', xy=(arrow_land_zb, arrow_land_price), xytext=(zb[4], entry),
        arrowprops=dict(
            arrowstyle='-|>', color=C_A, lw=3.0, mutation_scale=22,
            connectionstyle='arc3,rad=0.15' if is_bull else 'arc3,rad=-0.15'
        ), zorder=8)

    price_range = max(v) - min(v)
    label_offset = price_range * 0.08
    if is_bull:
        pct_y = arrow_land_price + label_offset
    else:
        pct_y = arrow_land_price - label_offset

    pct = ((target - entry) / entry) * 100
    ax.text(arrow_land_zb, pct_y, f'{pct:+.1f}%',
            fontsize=13, fontweight='bold', color=C_A,
            ha='center', va='bottom' if is_bull else 'top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C_A, alpha=0.9, lw=0.8),
            zorder=10)

    labels_txt = ['P1','P2','P3','P4','P5']
    for i in range(5):
        is_low = pts[i]['type'] == 'L'
        dt_str = pts[i]['date'].strftime('%b %d')
        ax.annotate(
            f'{labels_txt[i]}  {v[i]:.2f}\n{dt_str}',
            xy=(zb[i], v[i]),
            xytext=(0, -28 if is_low else 28),
            textcoords='offset points',
            ha='center', va='top' if is_low else 'bottom',
            fontsize=8.5, fontweight='bold', color=C_W,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=C_W, alpha=0.9, lw=0.6),
            arrowprops=dict(arrowstyle='-', color=C_W, lw=0.6))

    # Title with Arabic name
    emoji = '📈' if is_bull else '📉'
    title_text = (
        f'{emoji}   {ticker}  -  {ar_name_shaped}   |   '
        f'{direction_shaped}   |   {tf_shaped}'
    )
    ax.set_title(title_text,
                 fontsize=16, fontweight='bold', pad=16, color='#212121')
    ax.set_ylabel('')

    bc = '#E8F5E9' if is_bull else '#FFEBEE'
    bt = '#2E7D32' if is_bull else '#C62828'

    # Info box with Arabic labels
    lbl_direction = shape_arabic('موجة وولف ' + direction)
    lbl_name      = shape_arabic('السهم: ' + ar_name)
    lbl_close     = shape_arabic('آخر إغلاق')
    lbl_entry     = shape_arabic('دخول (P5)')
    lbl_target    = shape_arabic('هدف (1→4)')
    lbl_potential = shape_arabic('النسبة المتوقعة')
    lbl_tf        = shape_arabic('الإطار الزمني: ' + tf_label)

    info = (
        f"  {lbl_direction}\n"
        f"  ─────────────────────\n"
        f"  {lbl_name}\n"
        f"  {lbl_close} :  {last_close:.2f}\n"
        f"  {lbl_entry} :  {entry:.2f}\n"
        f"  {lbl_target} :  {target:.2f}\n"
        f"  {lbl_potential} :  {pct:+.1f}%\n"
        f"  {lbl_tf}"
    )

    ax.text(0.01, 0.03, info,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold', color=bt,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bc,
                      edgecolor=bt, alpha=0.92, lw=1.2),
            zorder=10)

    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────
# 7. TICKER PROCESSING
# ────────────────────────────────────────────────────────────

def process_ticker(ticker, period, interval, resample_rule=None):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or len(df) < 30:
            return ticker, [], None

        if resample_rule:
            df = resample_ohlc(df, resample_rule)
            if len(df) < 30:
                return ticker, [], None

        found = find_active_wolfe(df, max_bars_since_p5=8)
        last_bar = len(df) - 1
        for r in found:
            b1=r['points'][0]['bar']; v1=r['points'][0]['price']
            b4=r['points'][3]['bar']; v4=r['points'][3]['price']
            r['target_price'] = round(line_at(last_bar, b1,v1, b4,v4), 2)
            r['last_close']   = round(df[
