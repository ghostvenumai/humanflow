# Latenzbudget (Ziele, keine Messungen)

Diese Tabelle nennt **Zielwerte** pro Pipelinestufe, keine gemessenen Real-Call-
Latenzen. Die Werte unter `reports/*.json` sind lokal/simuliert und dürfen nicht als
Real-Call-Latenz gelesen werden.

Repo-verankerte Ziele stammen aus `config/quality-gates.yaml`; Timing- und
Degradations-Parameter aus den Konstruktor-Defaults von
`src/humanflow/runtime/session.py`.

| Stufe | Zielwert | Quelle |
|---|---|---|
| TTS: erstes Audio (`ttfa_ms`) | Ziel `< 700 ms` | `config/quality-gates.yaml` |
| Hörbares Barge-in (`audible_barge_in_latency_ms`) | Ziel `< 250 ms` | `config/quality-gates.yaml` |
| STT: Final nach Sprechende | kein repo-verankertes Ziel — Messung erforderlich | — |
| Reasoner: erstes Token | kein repo-verankertes Ziel — Messung erforderlich | — |

Jede Zahl oben ist ein **Zielwert**, keine Messung. Für die letzten beiden Stufen
existiert im Repo kein belastbarer Zielwert; sie sind bewusst offen gelassen statt
geschätzt.

## Degradation bei Budgetüberschreitung

Bleibt das TTS-Erstaudio aus, greift ein hartes Timeout: nach
`tts_first_audio_timeout_ms` (Default `6000 ms`) bricht der Sprech-Versuch ab und die
Session recovert deterministisch nach `LISTENING`, statt in `THINKING` zu stranden.
Ein transienter Fehler *vor* dem ersten Audio wird zuvor einmal mit demselben,
bereits erzeugten Text neu synthetisiert — ohne Tool- oder Reasoner-Wiederholung.

Weitere belegbare Timing-Defaults (`session.py`):

- `final_admission_reconciliation_ms = 60` (zulässiger Bereich `0–250`)
- `soft_yield_recovery_delay_ms = 420`

Die zuerst degradierende Stufe ist damit die **TTS-Erstaudio-Stufe**: sie ist die
einzige mit einem harten, im Code verankerten Timeout und einer definierten
deterministischen Recovery.
