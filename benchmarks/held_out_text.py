"""Held-out sentences for the estimator, with their real cl100k_base token counts.

Measured with tiktoken on 2026-11-08. They are NOT the text the per-script rates were fitted on,
so they show how the estimator does on text it has not seen. Used by tests/test_round5.py and
benchmarks/estimator_bench.py.
"""

HELD_OUT = [
    (
        "greek",
        "Ο υπολογιστής μου χάλασε χθες το βράδυ και δεν μπορώ να ανοίξω τα αρχεία της δουλειάς. Θα πρέπει να καλέσω τεχνικό αύριο το πρωί για να τον επισκευάσει.",
        130,
    ),
    (
        "russian",
        "Вчера вечером мой компьютер сломался, и я не могу открыть рабочие файлы. Завтра утром придётся вызвать мастера, чтобы он всё починил.",
        62,
    ),
    (
        "arabic",
        "تعطل حاسوبي مساء أمس ولا أستطيع فتح ملفات العمل. سيتعين علي استدعاء فني صباح الغد لإصلاحه.",
        64,
    ),
    (
        "hebrew",
        "המחשב שלי התקלקל אתמול בערב ואני לא יכול לפתוח את קבצי העבודה. מחר בבוקר אצטרך להזמין טכנאי.",
        90,
    ),
    (
        "hindi",
        "मेरा कंप्यूटर कल शाम खराब हो गया और मैं काम की फ़ाइलें नहीं खोल पा रहा हूँ। कल सुबह मुझे तकनीशियन को बुलाना होगा।",
        114,
    ),
    ("thai", "คอมพิวเตอร์ของฉันเสียเมื่อคืนนี้และฉันเปิดไฟล์งานไม่ได้ พรุ่งนี้เช้าฉันต้องเรียกช่างมาซ่อม", 85),
    (
        "german",
        "Der schnelle braune Fuchs springt über den faulen Hund, während der Ausschuss den Haushaltsentwurf für das nächste Geschäftsjahr prüft. Größere Änderungen müssen genehmigt werden.",
        53,
    ),
    (
        "spanish",
        "El zorro marrón rápido salta sobre el perro perezoso mientras el comité revisa el presupuesto trimestral. La información será enviada mañana por la tarde.",
        38,
    ),
]
