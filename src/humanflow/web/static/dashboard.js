async function main() {
const [scorecard, evidence] = await Promise.all([
  fetch("/api/scorecard").then((response) => response.json()),
  fetch("/api/evidence").then((response) => response.json())
]);

const format = (value) => value == null ? "—" : (typeof value === "number" ? Number(value.toFixed(3)).toString() : value);
const release = document.getElementById("release-status");
release.textContent = `${scorecard.summary.engineering_evidence_status} · Produktion: ${scorecard.summary.production_release_claim}`;

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
}

main().catch((error) => {
  const status = document.getElementById("release-status");
  status.textContent = `Dashboard-Fehler: ${error.name}`;
});
