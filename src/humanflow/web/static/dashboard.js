async function main() {
const [scorecard, evidence] = await Promise.all([
  fetch("/api/scorecard").then((response) => response.json()),
  fetch("/api/evidence").then((response) => response.json())
]);

const format = (value) => value == null ? "—" : (typeof value === "number" ? Number(value.toFixed(3)).toString() : value);
const release = document.getElementById("release-status");
release.textContent = `${scorecard.summary.engineering_evidence_status} · Produktion: ${scorecard.summary.production_release_claim}`;

const evidenceScopes = document.getElementById("evidence-scopes");
Object.entries(evidence.scopes).forEach(([name, value]) => {
  const card = document.createElement("article");
  card.className = "gate";
  const title = document.createElement("h3");
  title.textContent = name.replaceAll("_", " ");
  const detail = document.createElement("div");
  detail.className = "metric";
  detail.textContent = value;
  card.append(title, detail);
  evidenceScopes.append(card);
});

const gates = document.getElementById("gates");
Object.entries(scorecard.gates).forEach(([name, gate]) => {
  const metric = scorecard.metrics[name];
  const card = document.createElement("article");
  card.className = `gate ${gate.passed ? "pass" : "fail"}`;
  const title = document.createElement("h3");
  title.textContent = name;
  const current = document.createElement("div");
  current.className = "metric";
  current.textContent = format(gate.observed);
  const compare = document.createElement("div");
  compare.className = "compare";
  [["Baseline", metric.baseline_value], ["Aktuell", gate.observed], ["Ziel", `${gate.operator} ${gate.target}`]].forEach(([label, value]) => {
    const cell = document.createElement("div");
    const caption = document.createElement("span");
    caption.textContent = label;
    const detail = document.createElement("strong");
    detail.textContent = format(value);
    cell.append(caption, detail);
    compare.append(cell);
  });
  const scope = document.createElement("p");
  scope.className = "scope";
  scope.textContent = `n=${gate.sample_count} · ${gate.scope}`;
  card.append(title, current, compare, scope);
  gates.append(card);
});

document.getElementById("quality-loop").innerHTML = `
  <p class="big-result">${evidence.quality_loop.decision}</p>
  <p>Score ${evidence.quality_loop.baseline_score} → ${evidence.quality_loop.candidate_score}</p>
  <p class="note">Protected hashes: ${evidence.quality_loop.protected_hashes_equal ? "identisch" : "abweichend"}</p>`;
document.getElementById("torture").innerHTML = `
  <p class="big-result">${evidence.torture.passed}/${evidence.torture.total}</p>
  <p>T01–T20 bestanden · ${evidence.timeline.events} replayed Events</p>`;
document.getElementById("router").innerHTML = `
  <p class="big-result">${evidence.router.correct}/${evidence.router.total}</p>
  <p>Router-Matrix · Tournament ${evidence.tournament.status}</p>
  <p class="note">Externe Agent-Aufrufe: ${evidence.tournament.external_agent_calls_made}</p>`;
document.getElementById("provenance").innerHTML = `
  <p><strong>Start:</strong> ${evidence.sprint.start_utc}</p>
  <p><strong>Deadline:</strong> ${evidence.sprint.deadline_utc}</p>
  <p><strong>Tag:</strong> ${evidence.sprint.tag}</p>
  <p class="note">Baseline ${evidence.sprint.baseline_commit}</p>`;

const artifacts = document.getElementById("artifacts");
Object.entries(evidence.artifacts).forEach(([name, artifact]) => {
  const item = document.createElement("li");
  item.textContent = `${name} · ${artifact.sha256}`;
  artifacts.append(item);
});

document.getElementById("load-timeline").addEventListener("click", async () => {
  const timeline = await fetch("/api/timeline").then((response) => response.json());
  const list = document.getElementById("dashboard-events");
  list.replaceChildren();
  timeline.events.forEach((event) => {
    const item = document.createElement("li");
    item.textContent = `${event.conversation_id} · ${event.sequence} · ${event.event_type} · ${event.monotonic_ns}`;
    list.append(item);
  });
});

const liveLabels = {
  acoustic_speech_onset_latency_ms: "Akustische Onset-Erkennung",
  speech_onset_to_soft_duck_ms: "Soft Yield",
  speech_onset_to_hard_cancel_ms: "Bestätigte Unterbrechung",
  speech_onset_to_audible_stop_ms: "Audible Stop ACK",
  backchannel_recovery_latency_ms: "Backchannel Recovery",
  first_stt_partial_ms: "Erstes STT-Partial",
  final_stt_ms: "Finales STT"
};

async function refreshLiveBargeIn() {
  const live = await fetch("/api/live-barge-in").then((response) => response.json());
  const panel = document.getElementById("live-barge-in");
  panel.replaceChildren();
  Object.entries(live.metrics).forEach(([name, metric]) => {
    const card = document.createElement("article");
    card.className = "gate";
    const title = document.createElement("h3");
    title.textContent = liveLabels[name] || name;
    const value = document.createElement("div");
    value.className = "metric";
    value.textContent = metric.last == null ? "—" : `${format(metric.last)} ms`;
    const scope = document.createElement("p");
    scope.className = "scope";
    scope.textContent = `n=${metric.sample_count}`;
    card.append(title, value, scope);
    panel.append(card);
  });
  const falseCard = document.createElement("article");
  falseCard.className = `gate ${live.false_interruption_count ? "fail" : "pass"}`;
  falseCard.innerHTML = `<h3>Falsche Hard-Cancels</h3><div class="metric">${live.false_interruption_count}</div><p class="scope">laufende Session</p>`;
  panel.append(falseCard);
  document.getElementById("live-barge-in-scope").textContent = `${live.status} · ${live.measurement_scope}`;
}

await refreshLiveBargeIn();
window.setInterval(() => refreshLiveBargeIn().catch(() => {}), 1000);

function money(cost) {
  if (!cost || cost.value == null) return "Kosten nicht verfügbar";
  const value = Number(cost.value);
  const currency = cost.currency || "";
  const digits = Math.abs(value) >= 1 ? 2 : (Math.abs(value) >= 0.01 ? 4 : 6);
  return `${currency} ${value.toFixed(digits)}`.trim();
}

async function refreshEconomics() {
  const payload = await fetch("/api/costs").then((response) => response.json());
  const panel = document.getElementById("session-economics");
  const providers = document.getElementById("provider-economics");
  panel.replaceChildren();
  providers.replaceChildren();
  if (!payload.summary) {
    document.getElementById("economics-scope").textContent = `${payload.status} · LOCAL DEMO SQLITE`;
    return;
  }
  const summary = payload.summary;
  const values = [
    ["Dauer", summary.active_duration_seconds == null ? "—" : `${format(Number(summary.active_duration_seconds))} s`],
    ["Turns", summary.turn_count],
    ["STT", summary.services.STT?.audio_input_seconds ?? "0 s"],
    ["LLM", `${summary.services.LLM?.input_tokens || 0} in / ${summary.services.LLM?.output_tokens || 0} out`],
    ["TTS erzeugt", `${summary.played_audio_economics.generated_audio_seconds} s`],
    ["TTS ungehört", `${summary.played_audio_economics.unheard_audio_seconds} s`],
    ["Tools", `${summary.tools.successful_actions}/${summary.tools.call_count} erfolgreich`],
    ["Provider gemeldet", money(summary.provider_reported_cost)],
    ["Geschätzt", money(summary.estimated_cost)],
    ["Kosten/Turn", money(summary.cost_per_turn)],
    ["Kosten/Minute", money(summary.cost_per_conversation_minute)],
    ["Barge-in-Waste", summary.played_audio_economics.wasted_cost_estimate]
  ];
  values.forEach(([name, value]) => {
    const card = document.createElement("article");
    card.className = "gate";
    const title = document.createElement("h3");
    title.textContent = name;
    const detail = document.createElement("div");
    detail.className = "metric";
    detail.textContent = value;
    card.append(title, detail);
    panel.append(card);
  });
  const heading = document.createElement("h3");
  heading.textContent = "Provider-Aufschlüsselung";
  const list = document.createElement("ul");
  summary.providers.forEach((provider) => {
    const item = document.createElement("li");
    const percentage = provider.estimated_cost_percentage_known == null
      ? "Anteil nicht verfügbar"
      : `${format(Number(provider.estimated_cost_percentage_known))} % des bekannten Schätzwerts`;
    item.textContent = `${provider.provider} · ${provider.model} · ${provider.service} · n=${provider.event_count} · ${percentage} · ${provider.evidence.join(" + ")}`;
    list.append(item);
  });
  providers.append(heading, list);
  document.getElementById("economics-scope").textContent = `REAL_BROWSER_SESSION · ${summary.evidence_labels.join(" + ")} · PRODUCTION TELEPHONY NOT ESTABLISHED`;
}

await refreshEconomics();
window.setInterval(() => refreshEconomics().catch(() => {}), 3000);
}

main().catch((error) => {
  const status = document.getElementById("release-status");
  status.textContent = `Dashboard-Fehler: ${error.name}`;
});
