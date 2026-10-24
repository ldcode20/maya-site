// Everyday-user app: one payment, checked by Maya before it leaves.
// Optionally connected to a judges' demo screen (?code=XXXXXX or Settings):
// the server then shows each decision on that screen too.
import { getScenarios, joinRoom, rupees, score, signalGroups, store } from "/static/api.js";
import { DEFAULT_ANSWERS, customScenario } from "/static/custom.js";
import { REASONS, initialLang, rememberLang, t } from "/static/i18n.js";

const $ = (sel) => document.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const params = new URLSearchParams(location.search);
const MIN_CHECK_MS = 700;  // let the dot visibly travel before the answer lands
const ROOM_KEY = "maya-room";
const CUSTOM_KEY = "maya-custom";  // the last "Your own payment" answers

let lang = initialLang();
let scenarios = [];
let current = null;        // scenario object from /v1/demo/scenarios
let result = null;         // last /v1/score response
let view = "pay";          // pay | checking | outcome | nudge-yes | calling | cancelled | sent | error
let errorKind = "network";
let confirmedAfterWarning = false;
let timer = null, holdEndsAt = 0;
let room = store.get(ROOM_KEY);   // { publish_token, code } when connected to a demo screen

// --- helpers -------------------------------------------------------------------

const L = () => t(lang);
const pick = (obj) => (obj && typeof obj === "object" ? obj[lang] ?? obj.en : obj);
const money = (amt) => rupees(amt);
// From the English name: first and last word, ignoring "(…)" and abbreviations
// like "R.". A name in another script (typed into "Your own payment") shows its
// first whole character instead: Devanagari letters can't be split into initials.
const firstChar = (w) => (window.Intl?.Segmenter ? [...new Intl.Segmenter().segment(w)][0]?.segment : w[0]) ?? "?";
const initials = (payee) => {
  const words = String(payee?.en ?? payee).replace(/\(.*?\)/g, "").trim().split(/\s+/).filter((w) => w && !w.endsWith("."));
  if (!words.length) return "?";
  if (!/^[A-Za-z0-9]/.test(words[0])) return firstChar(words[0]);
  return (words.length > 1 ? words[0][0] + words[words.length - 1][0] : words[0][0]).toUpperCase();
};

function amountMarkup(amt) {
  const text = money(amt);
  return `<span class="sym" aria-hidden="true">₹</span>${esc(text.replace("₹", ""))}<span class="visually-hidden"> rupees</span>`;
}

/** Plain-language reasons: model reasons first, then the phone's own signals. */
function reasonList() {
  const groups = signalGroups(result);
  const keys = [...(result.reason_features ?? [])];
  if (groups.receiving.includes("mulehunter")) keys.push("mulehunter");
  for (const k of groups.phone) keys.push(k);
  return [...new Set(keys)].map((k) => REASONS[lang][k]).filter(Boolean);
}

function phoneSignals() {
  const soft = current?.soft_signals ?? {};
  const items = [];
  if (soft.is_screen_shared) items.push(L().phone_screen);
  if (soft.is_voice_active) items.push(L().phone_voice);
  if (soft.chat_stage === "payment_demand" || soft.chat_stage === "escalated") items.push(L().phone_chat);
  return items;
}

function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

function tick() {
  const el = $("#timer");
  if (!el) return stopTimer();
  const left = Math.max(0, holdEndsAt - Date.now());
  const m = Math.floor(left / 60000), s = Math.floor((left % 60000) / 1000);
  el.textContent = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// --- demo-screen connection ------------------------------------------------------

function renderLive() {
  const live = $("#live");
  live.hidden = !room;
  live.textContent = room ? L().connected(room.code) : "";
  $("#disconnect").hidden = !room;
}

async function connect(code) {
  const status = $("#connect-status");
  try {
    const joined = await joinRoom(code);
    room = { publish_token: joined.publish_token, code: joined.code };
    store.set(ROOM_KEY, room);
    status.textContent = L().connected(room.code);
    status.classList.remove("error");
  } catch {
    status.textContent = L().connect_failed;
    status.classList.add("error");
  }
  renderLive();
}

function disconnect(note = false) {
  room = null;
  store.drop(ROOM_KEY);
  const status = $("#connect-status");
  status.textContent = note ? L().disconnected_note : L().connect_hint;
  status.classList.toggle("error", note);
  renderLive();
}

// --- rendering --------------------------------------------------------------------

function flight(state) {
  const label = view === "checking" ? L().checking : "";
  return `<div class="flight" data-state="${state}" aria-hidden="true">
      <span>${lang === "hi" ? "आप" : "You"}</span>
      <div class="track"><span class="run"></span><span class="dot"></span></div>
      <span>${esc(initials(current.payee))}</span>
      <span class="checking-label">${esc(label)}</span>
    </div>`;
}

function header() {
  return `<div class="payee">
      <div class="avatar" aria-hidden="true">${esc(initials(current.payee))}</div>
      <div><p class="payee-name"><span class="visually-hidden">${esc(L().paying)} </span>${esc(pick(current.payee))}</p>
        <p class="payee-upi">${esc(current.upi)}</p></div>
    </div>
    <p class="amount">${amountMarkup(current.amount_inr)}</p>
    <p class="note">${esc(pick(current.note))}</p>
    <p class="from">${esc(L().from_account)}</p>`;
}

/** Up to 3 reasons in view; anything else (extra reasons, `extra` HTML) goes
 *  into a single "More details" disclosure. */
function reasonsBlock(extra = "") {
  const all = reasonList();
  const shown = all.slice(0, 3), rest = all.slice(3);
  const list = shown.length ? `<ul class="reasons" aria-label="${esc(L().why)}">${shown.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";
  const more = rest.length || extra
    ? `<details class="more"><summary>${esc(L().more_details)}</summary>
        ${rest.length ? `<ul class="reasons">${rest.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}${extra}</details>`
    : "";
  return list + more;
}

function outcomePanel() {
  const helpline = `<p class="helpline">${esc(L().helpline).replace("1930", '<a href="tel:1930">1930</a>')}</p>`;
  if (result.action === "hold") {
    return `<section class="outcome hold" aria-labelledby="o-title">
        <h2 id="o-title" tabindex="-1">${esc(L().hold_title(result.auto_release_minutes || 60))}</h2>
        <time class="timer" id="timer" role="timer" aria-live="off">60:00</time>
        <p>${esc(L().hold_body)}</p>
        ${reasonsBlock(`<p>${esc(result.requires_analyst ? L().hold_analyst : L().hold_auto)}</p>${helpline}`)}
      </section>
      <div class="actions">
        <button class="pill" data-act="call">${esc(L().call_trusted)}</button>
        <button class="pill ghost" data-act="cancel">${esc(L().cancel_payment)}</button>
      </div>`;
  }
  if (result.action === "nudge") {
    return `<section class="outcome nudge" aria-labelledby="o-title">
        <h2 id="o-title" tabindex="-1">${esc(L().nudge_title)}</h2>
        <p class="question">${esc(L().nudge_question)}</p>
        ${reasonsBlock()}
      </section>
      <div class="actions">
        <button class="pill" data-act="yes">${esc(L().nudge_yes)}</button>
        <button class="pill ghost" data-act="no">${esc(L().nudge_no)}</button>
      </div>`;
  }
  return "";  // release goes straight to the receipt
}

function receipt() {
  const ref = (result?.request_id ?? "").replace(/[^0-9a-f]/gi, "").slice(0, 12).toUpperCase();
  const when = new Date().toLocaleString(lang === "hi" ? "hi-IN" : "en-IN",
    { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Kolkata" });
  return `<section class="receipt" aria-labelledby="r-title">
      <div class="tick" aria-hidden="true"><svg viewBox="0 0 24 24" width="30" height="30"><path d="M5 12.5 10 17l9-10" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
      <h2 id="r-title" tabindex="-1">${esc(L().sent_title(money(current.amount_inr), pick(current.payee)))}</h2>
      <dl>
        <div><dt>${lang === "hi" ? "समय" : "When"}</dt><dd>${esc(when)}</dd></div>
        <div><dt>UPI ID</dt><dd>${esc(current.upi)}</dd></div>
        <div><dt>${esc(L().ref)}</dt><dd>${esc(ref)}</dd></div>
      </dl>
      <p class="checked">${esc(confirmedAfterWarning ? L().sent_after_warning : L().sent_checked(Math.max(1, Math.round(result?.latency_ms ?? 0))))}</p>
    </section>
    <div class="actions"><button class="pill ghost" data-act="restart">${esc(L().start_over)}</button></div>`;
}

function render() {
  const screen = $("#screen");
  const status = $("#phone-status");
  const signals = current ? phoneSignals() : [];
  status.hidden = !signals.length || ["sent", "cancelled", "error"].includes(view);
  status.innerHTML = signals.map((s) => `<li>${esc(s)}</li>`).join("");

  if (view === "error") {
    stopTimer();
    screen.innerHTML = `<section class="outcome neutral" role="alert"><h2>${esc(L().error_title)}</h2>
      <p>${esc(errorKind === "models" ? L().error_models : L().error_network)}</p></section>`;
    return;
  }
  if (!current) return;
  if (view === "sent") { stopTimer(); screen.innerHTML = receipt(); return; }
  if (view === "cancelled") {
    stopTimer();
    screen.innerHTML = `<section class="outcome neutral" aria-labelledby="c-title"><h2 id="c-title" tabindex="-1">${esc(L().cancelled_title)}</h2>
      <p>${esc(L().cancelled_body)}</p></section>
      <div class="actions"><button class="pill ghost" data-act="restart">${esc(L().start_over)}</button></div>`;
    return;
  }
  if (view === "calling") {
    screen.innerHTML = `<section class="calling" aria-live="polite"><div class="ring" aria-hidden="true">P</div>
      <h2>${esc(L().calling)}</h2><p class="helpline">${esc(L().calling_body)}</p></section>
      <div class="actions"><button class="pill ghost" data-act="end-call">${esc(L().end_call)}</button></div>`;
    return;
  }
  if (view === "nudge-yes") {
    screen.innerHTML = header() + `<section class="outcome hold" role="alert"><h2>${esc(L().nudge_title)}</h2>
      <p>${esc(L().nudge_stop).replace("1930", '<a href="tel:1930">1930</a>')}</p></section>
      <div class="actions"><button class="pill" data-act="cancel">${esc(L().cancel_payment)}</button>
      <a class="pill ghost" href="tel:1930">1930</a></div>`;
    return;
  }

  const state = view === "checking" ? "checking" : view === "outcome" ? result.action : "idle";
  const body = view === "pay" || view === "checking"
    ? `<div class="actions"><button class="pill" data-act="pay" ${view === "checking" ? "disabled" : ""}>${esc(L().pay(money(current.amount_inr)))}</button></div>`
    : outcomePanel();
  // The line is created where it was, then moved on the next frame, so the dot
  // visibly travels: idle -> checking on Pay, checking -> outcome on the answer.
  screen.innerHTML = header() + flight(view === "outcome" ? "checking" : "idle") + body;
  if (state !== "idle") {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const f = screen.querySelector(".flight");
      if (f) f.dataset.state = state;
    }));
  }
  if (view === "outcome" && result.action === "hold") {
    if (!holdEndsAt) holdEndsAt = Date.now() + (result.auto_release_minutes || 60) * 60000;
    tick();
    if (!timer) timer = setInterval(tick, 1000);
  }
}

// --- actions --------------------------------------------------------------------

async function pay() {
  view = "checking";
  render();
  const started = performance.now();
  try {
    result = await score(current, Boolean(room), room ? {
      room: room.publish_token,
      context: { scenario: current.key, payee: current.payee.en, note: current.note.en },
    } : {});
  } catch (err) {
    errorKind = err.status === 503 ? "models" : "network";
    view = "error";
    render();
    return;
  }
  if (room && result.room_published === false) disconnect(true);
  const wait = MIN_CHECK_MS - (performance.now() - started);
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  holdEndsAt = 0;
  view = "outcome";
  render();
  if (result.action === "release") setTimeout(() => { if (view === "outcome") { view = "sent"; render(); $("#r-title")?.focus(); } }, 1100);
  else $("#o-title")?.focus();
}

function loadScenario(sc) {
  current = sc;
  result = null; view = "pay"; confirmedAfterWarning = false; holdEndsAt = 0;
  stopTimer();
  render();
}

function load(key) {
  const saved = store.get(CUSTOM_KEY);
  if (key === "custom" && saved) return loadScenario(customScenario(saved));
  loadScenario(scenarios.find((s) => s.key === key) ?? scenarios[0]);
}

// --- "Your own payment" form --------------------------------------------------------

function openCustom() {
  const a = { ...DEFAULT_ANSWERS, ...(store.get(CUSTOM_KEY) ?? {}) };
  const form = $("#custom-form");
  for (const [k, v] of Object.entries(a)) {
    const el = form.elements.namedItem(k);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = Boolean(v); else el.value = String(v);
  }
  $("#custom-error").hidden = true;
  $("#settings").close();
  $("#custom").showModal();
  form.elements.namedItem("payee").focus();
}

$("#custom-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const form = e.target;
  const a = {};
  for (const k of Object.keys(DEFAULT_ANSWERS)) {
    const el = form.elements.namedItem(k);
    if (!el) continue;
    a[k] = el.type === "checkbox" ? el.checked : el.type === "number" || el.tagName === "SELECT" ? Number(el.value) : el.value.trim();
  }
  const ok = a.payee && Number.isFinite(a.amount) && a.amount >= 1 && a.amount <= 10_000_000
    && Number.isFinite(a.usual) && a.usual >= 1;
  if (!ok) {
    const err = $("#custom-error");
    err.textContent = L().custom_error;
    err.hidden = false;
    return;
  }
  store.set(CUSTOM_KEY, a);
  $("#custom").close();
  loadScenario(customScenario(a));
  $("#screen").focus();
});

function setLang(next) {
  lang = next;
  rememberLang(lang);
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  document.querySelectorAll("[data-i18n]").forEach((el) => { const s = L()[el.dataset.i18n]; if (typeof s === "string") el.textContent = s; });
  document.title = `${L().app_title} · Maya`;
  renderScenarioList();
  renderLive();
  render();
}

function renderScenarioList() {
  $("#scenario-list").innerHTML = scenarios.map((s) =>
    `<li><button type="button" data-key="${esc(s.key)}"><strong>${esc(pick(s.label))}</strong><span>${esc(money(s.amount_inr))}</span></button></li>`).join("");
}

document.addEventListener("click", (e) => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  const langBtn = e.target.closest("[data-lang]");
  const pickBtn = e.target.closest("#scenario-list [data-key]");
  if (langBtn) { setLang(langBtn.dataset.lang); return; }
  if (pickBtn) { $("#settings").close(); load(pickBtn.dataset.key); return; }
  if (e.target.closest("#open-settings")) { $("#settings").showModal(); return; }
  if (e.target.closest("#open-custom")) { openCustom(); return; }
  if (e.target.closest("#custom-close")) { $("#custom").close(); return; }
  if (e.target.closest("#disconnect")) { disconnect(); return; }
  if (!act) return;
  if (act === "pay") pay();
  if (act === "cancel") { view = "cancelled"; render(); $("#c-title")?.focus(); }
  if (act === "call") { view = "calling"; render(); }
  if (act === "end-call") { view = "outcome"; render(); }
  if (act === "yes") { view = "nudge-yes"; render(); }
  if (act === "no") {
    confirmedAfterWarning = true;
    const btn = e.target.closest("[data-act]");
    btn.dataset.act = "pay-anyway";
    btn.textContent = L().pay_anyway(money(current.amount_inr));
  }
  if (act === "pay-anyway") { view = "sent"; render(); $("#r-title")?.focus(); }
  if (act === "restart") load(current.key);
});

$("#connect-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const code = $("#code-input").value.trim();
  if (code) connect(code);
});

// --- start -------------------------------------------------------------------------

(async function start() {
  const qrCode = params.get("code");
  if (qrCode) {
    history.replaceState(null, "", location.pathname + (params.get("s") ? `?s=${encodeURIComponent(params.get("s"))}` : ""));
  }
  try {
    scenarios = await getScenarios();
  } catch {
    view = "error"; errorKind = "network";
    setLang(lang);
    return;
  }
  setLang(lang);
  load(params.get("s") ?? "police");
  if (qrCode) await connect(qrCode);
})();
