"""Bilingual (English/Khmer) HTML-safe Telegram message templates."""
from __future__ import annotations

import html
from typing import Any

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "disclaimer": DISCLAIMER,
        "welcome": (
            "Welcome to Meas Pro Alerts, {name}!\n\n"
            "You will receive every new BUY/SELL signal and its TP/SL updates "
            "automatically.\nUse /settings to choose grade, timeframes and language.\n\n"
            "Commands:\n/start /stop /help /signal /price /settings /history /stats /levels /lang"
        ),
        "stopped": "Alerts paused. Use /start to resume.",
        "help": (
            "Commands:\n"
            "/start - subscribe\n/stop - pause alerts\n"
            "/signal - latest signal\n/price - live XAU/USD\n"
            "/settings - grade, timeframes, language, pause\n"
            "/history - last 10 signals\n/stats - win rate and profit factor\n"
            "/levels - key support/resistance\n/lang - language\n"
            "/link CODE - connect your website account"
        ),
        "why": "Why:",
        "invalid_if": "Invalid if:",
        "entry": "Entry",
        "stop_loss": "Stop loss",
        "tp1": "TP1",
        "tp2": "TP2",
        "tp3": "TP3",
        "risk_reward": "Risk:Reward",
        "confidence": "Confidence",
        "grade": "Grade",
        "open_chart": "Open chart",
        "delayed": "DELAYED (sent after wake-up)",
        "no_signal": "No signal available yet.",
        "latest_signal": "Latest signal:",
        "price": "XAU/USD: {price}",
        "price_unavailable": "Live price is temporarily unavailable.",
        "history": "Last {count} signals:",
        "history_empty": "No signals recorded yet.",
        "stats": "Signals: {total} | Closed: {closed}\nWin rate: {win_rate}% | Profit factor: {profit_factor}",
        "levels": "Key levels ({tf}):\n{levels}",
        "levels_empty": "No active levels right now.",
        "settings_title": "Your settings\nGrade: {grade}\nTimeframes: {timeframes}\nLanguage: {language}\nStatus: {status}",
        "min_grade": "Min grade",
        "timeframes": "Timeframes",
        "language": "Language",
        "pause": "Pause alerts",
        "resume": "Resume alerts",
        "settings_saved": "Setting updated: {key} = {value}",
        "paused": "Alerts paused.",
        "resumed": "Alerts resumed.",
        "lang_changed": "Language set to English.",
        "linked": "Website account connected.",
        "link_invalid": "Invalid or expired link code. Generate a new one on the Settings page.",
        "not_admin": "This command is for administrators only.",
        "broadcast_usage": "Usage: /broadcast <message>",
        "broadcast_done": "Broadcast sent to {count} subscribers.",
        "users_count": "Subscribers: {count} (active)",
        "status": (
            "Bot: @{username}\nWebhook: {webhook}\nSubscribers: {count}\n"
            "Last signal: {last_signal}\nData feed: {feed}\nErrors: {errors}"
        ),
        "test_signal": "Test signal delivered.",
        "tp_hit": "TP{hit} hit on {signal}.",
        "sl_hit": "Stop loss hit on {signal}.",
        "entry_zone": "Price entered the entry zone on {signal}.",
        "invalidated": "Signal {signal} invalidated.",
        "breakeven": "Stop loss moved to break-even on {signal}.",
        "quiet": "Quiet hours active. Message skipped.",
        "cap": "Daily message cap reached.",
        "unknown": "Unknown command. Use /help.",
        "signal_label": "{action} {symbol} #{id}",
    },
    "km": {
        "disclaimer": (
            "ការវិភាគអប់រំតែប៉ុណ្ណោះ។ មិនមែនជាដំបូន្មានហិរញ្ញវត្ថុទេ។ "
            "សញ្ញាអាចខុស។ កុំប្រថុយលុយដែលអ្នកមិនអាចចាញ់។"
        ),
        "welcome": (
            "សូមស្វាគមន៍មកកាន់ Meas Pro Alerts, {name}!\n\n"
            "អ្នកនឹងទទួលសញ្ញា BUY/SELL ថ្មីទាំងអស់ និងការអាប់ដេត TP/SL "
            "ដោយស្វ័យប្រវត្តិ។\nប្រើ /settings ដើម្បីជ្រើសថ្នាក់ ក្របខណ្ឌពេលវេលា និងភាសា។\n\n"
            "ពាក្យបញ្ជា៖\n/start /stop /help /signal /price /settings /history /stats /levels /lang"
        ),
        "stopped": "បានផ្អាកការព្រមាន។ ប្រើ /start ដើម្បីបន្ត។",
        "help": (
            "ពាក្យបញ្ជា៖\n"
            "/start - ចុះឈ្មោះ\n/stop - ផ្អាកការព្រមាន\n"
            "/signal - សញ្ញាចុងក្រោយ\n/price - តម្លៃ XAU/USD ផ្ទាល់\n"
            "/settings - ថ្នាក់ ក្របខណ្ឌពេលវេលា ភាសា ផ្អាក\n"
            "/history - សញ្ញា ១០ ចុងក្រោយ\n/stats - អត្រាឈ្នះ និង profit factor\n"
            "/levels - កម្រិតគាំទ្រ/ទប់ទល់\n/lang - ភាសា\n"
            "/link CODE - ភ្ជាប់គណនីគេហទំព័រ"
        ),
        "why": "មូលហេតុ៖",
        "invalid_if": "មិនត្រឹមត្រូវប្រសិន៖",
        "entry": "ចូល",
        "stop_loss": "Stop loss",
        "tp1": "TP1",
        "tp2": "TP2",
        "tp3": "TP3",
        "risk_reward": "ហានិភ័យ៖រង្វាន់",
        "confidence": "ទំនុកចិត្ត",
        "grade": "ថ្នាក់",
        "open_chart": "បើកក្រាហ្វ",
        "delayed": "យឺត (ផ្ញើក្រោយភ្ញាក់)",
        "no_signal": "មិនទាន់មានសញ្ញាទេ។",
        "latest_signal": "សញ្ញាចុងក្រោយ៖",
        "price": "XAU/USD: {price}",
        "price_unavailable": "តម្លៃផ្ទាល់មិនទាន់មាន។",
        "history": "សញ្ញា {count} ចុងក្រោយ៖",
        "history_empty": "មិនទាន់មានសញ្ញាកត់ត្រាទេ។",
        "stats": "សញ្ញា៖ {total} | បិទ៖ {closed}\nអត្រាឈ្នះ៖ {win_rate}% | Profit factor៖ {profit_factor}",
        "levels": "កម្រិតសំខាន់ ({tf})៖\n{levels}",
        "levels_empty": "មិនមានកម្រិតសកម្មឥឡូវទេ។",
        "settings_title": (
            "ការកំណត់របស់អ្នក\nថ្នាក់៖ {grade}\nក្របខណ្ឌពេលវេលា៖ {timeframes}\n"
            "ភាសា៖ {language}\nស្ថានភាព៖ {status}"
        ),
        "min_grade": "ថ្នាក់អប្បបរមា",
        "timeframes": "ក្របខណ្ឌពេលវេលា",
        "language": "ភាសា",
        "pause": "ផ្អាកការព្រមាន",
        "resume": "បន្តការព្រមាន",
        "settings_saved": "បានធ្វើបច្ចុប្បន្នភាព៖ {key} = {value}",
        "paused": "បានផ្អាកការព្រមាន។",
        "resumed": "បានបន្តការព្រមាន។",
        "lang_changed": "បានកំណត់ភាសាជាខ្មែរ។",
        "linked": "បានភ្ជាប់គណនីគេហទំព័រ។",
        "link_invalid": "កូដភ្ជាប់មិនត្រឹមត្រូវ ឬផុតកំណត់។ សូមបង្កើតថ្មីនៅទំព័រកំណត់។",
        "not_admin": "ពាក្យបញ្ជានេះសម្រាប់អ្នកគ្រប់គ្រងតែប៉ុណ្ណោះ។",
        "broadcast_usage": "របៀបប្រើ៖ /broadcast <សារ>",
        "broadcast_done": "បានផ្ញើទៅអ្នកជាវ {count} នាក់។",
        "users_count": "អ្នកជាវ៖ {count} (សកម្ម)",
        "status": (
            "Bot៖ @{username}\nWebhook៖ {webhook}\nអ្នកជាវ៖ {count}\n"
            "សញ្ញាចុងក្រោយ៖ {last_signal}\nប្រភពទិន្នន័យ៖ {feed}\nកំហុស៖ {errors}"
        ),
        "test_signal": "បានផ្ញើសញ្ញាសាកល្បង។",
        "tp_hit": "TP{hit} បានដល់លើ {signal}។",
        "sl_hit": "Stop loss បានដល់លើ {signal}។",
        "entry_zone": "តម្លៃបានចូលតំបន់ចូលលើ {signal}។",
        "invalidated": "សញ្ញា {signal} មិនត្រឹមត្រូវ។",
        "breakeven": "បានផ្លាស់ទី Stop loss ទៅ break-even លើ {signal}។",
        "quiet": "ម៉ោងស្ងាត់។ មិនបានផ្ញើសារ។",
        "cap": "បានដល់ចំនួនសារអតិបរមាប្រចាំថ្ងៃ។",
        "unknown": "ពាក្យបញ្ជាមិនស្គាល់។ ប្រើ /help។",
        "signal_label": "{action} {symbol} #{id}",
    },
}


def lang_of(user: dict[str, Any] | None) -> str:
    return "km" if user and user.get("language") == "km" else "en"


def tr(language: str, key: str, **kwargs: Any) -> str:
    template = TEXT.get(language, TEXT["en"]).get(key) or TEXT["en"].get(key) or key
    return template.format(**kwargs) if kwargs else template


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def with_disclaimer(text: str, language: str) -> str:
    return f"{text}\n\n⚠️ {tr(language, 'disclaimer')}"


def format_signal(
    signal: dict[str, Any], language: str = "en", *, delayed: bool = False,
) -> str:
    action = str(signal.get("action") or "HOLD").upper()
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    explanation = signal.get("explanation") or {}
    if isinstance(explanation, str):
        explanation = {}
    reasons: list[tuple[bool, str]] = []
    agreeing = explanation.get("agreeing") or []
    disagreeing = explanation.get("disagreeing") or []
    for item in agreeing[:5]:
        reasons.append((True, str(item)))
    for item in disagreeing[:3]:
        reasons.append((False, str(item)))
    if not reasons:
        for item in (explanation.get("reasons") or [])[:6]:
            text = str(item)
            negative = any(
                token in text.lower()
                for token in ("rising", "fall", "bearish", "dxy", "weak", "not ", "risk")
            )
            reasons.append((not negative, text))
    lines = [
        f"{emoji} {esc(action)} {esc(signal.get('symbol') or 'XAU/USD')}  |  "
        f"{tr(language, 'grade')} {esc(signal.get('grade') or 'C')}  |  "
        f"{esc(signal.get('timeframe') or '')}",
    ]
    if delayed:
        lines.append(esc(tr(language, "delayed")))
    lines.extend([
        f"{tr(language, 'entry')}: {money(signal.get('entry'))}",
        f"{tr(language, 'stop_loss')}: {money(signal.get('stop_loss'))}",
        f"{tr(language, 'tp1')}: {money(signal.get('take_profit1'))}  |  "
        f"{tr(language, 'tp2')}: {money(signal.get('take_profit2'))}  |  "
        f"{tr(language, 'tp3')}: {money(signal.get('take_profit3'))}",
        f"{tr(language, 'risk_reward')} 1:{money(signal.get('risk_reward'))}  |  "
        f"{tr(language, 'confidence')} {money(signal.get('confidence'))}%",
        "",
        tr(language, "why"),
    ])
    if reasons:
        for positive, reason in reasons:
            lines.append(f"{'✅' if positive else '❌'} {esc(reason)}")
    else:
        lines.append(esc(explanation.get("summary") or ""))
    invalidation = explanation.get("invalidation")
    if invalidation:
        lines.extend(["", f"{tr(language, 'invalid_if')} {esc(invalidation)}"])
    return with_disclaimer("\n".join(lines), language)


def format_result(
    kind: str, signal: dict[str, Any], language: str, *, hit: str | None = None,
) -> str:
    label = tr(
        language, "signal_label",
        action=esc(signal.get("action") or ""),
        symbol=esc(signal.get("symbol") or "XAU/USD"),
        id=esc(signal.get("id") or ""),
    )
    key = {
        "tp": "tp_hit", "sl": "sl_hit", "entry": "entry_zone",
        "invalidated": "invalidated", "breakeven": "breakeven",
    }.get(kind, kind)
    body = tr(language, key, signal=label, hit=esc(hit or ""))
    return with_disclaimer(body, language)


def chart_markup(chart_url: str, language: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [[{
            "text": tr(language, "open_chart"), "url": chart_url,
        }]],
    }


def settings_markup(user: dict[str, Any], language: str) -> dict[str, Any]:
    def toggle(label: str, callback: str, enabled: bool) -> dict[str, str]:
        return {"text": f"{'✅' if enabled else '⬜'} {label}", "callback_data": callback}

    grade_buttons = [
        toggle(f"G:{grade}", f"tg:g:{grade}", str(user.get("min_grade")) == grade)
        for grade in ("A+", "A", "B", "C")
    ]
    tf_buttons = [
        toggle(tf, f"tg:t:{tf}", tf in str(user.get("timeframes") or "").split(","))
        for tf in ("15m", "1h", "4h", "1D")
    ]
    active = bool(user.get("is_active"))
    return {
        "inline_keyboard": [
            grade_buttons,
            tf_buttons,
            [
                toggle(tr(language, "language"), "tg:lang", True),
                toggle(tr(language, "resume" if not active else "pause"), "tg:pause", active),
            ],
        ],
    }
