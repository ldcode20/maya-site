// "Your own payment": turn plain answers into Maya's 27 signals.
//
// People can't type log-ratios, so the form asks plain questions and this maps
// them with the same formulas as src/features.py, starting from a *typical
// normal payment* in the training data (test-split legit medians) and changing
// only what each answer is about:
//   z_t   = log(amount / your usual payment)
//   e_t   = amount / usual            (first-time payee; else typical 0.01)
//   v_t   = earlier payments to this payee in 24 h (a regular contact starts at
//           the data's typical 12; a first-time payee at 0), + "earlier today"
//   zc_t  = log((amount + those payments) / usual)
//   d_R   = hops from the payee to a reported account (3 = none known)
//   ptr_t = how much of its inflow the payee passes on (0.95 if it forwards
//           money at once, else the typical 0.43)
// Payer-side answers go to *_snd, payee-side answers to *_rcv, and the combined
// columns use features.py's rules (max / min / larger |z|).
// Synthetic-data caveat: "normal" here means normal in the generated data.

const BASES = ["n_t", "e_t", "g_t", "z_t", "pi_t", "d_R", "v_t", "zc_t", "ptr_t"];
const REGULAR_V = 12;        // typical legit same-payee payments in 24 h
const REGULAR_PAIR = 21.4;   // typical legit same-payee 24 h total, in "usual" units

export const DEFAULT_ANSWERS = {
  payee: "", upi: "", amount: 2000, note: "",
  usual: 2000,            // your usual payment size (₹)
  firstTime: false,       // first time paying this person
  earlierToday: 0,        // earlier payments to them in the last 24 h
  probes: 0,              // tiny ₹1–10 test payments to them in the last hour
  manyNew: false,         // you've paid many new people this week
  link: 3,                // payee's link to a reported account: 0 reported, 1, 2 hops, 3 none known
  forwards: false,        // payee passes money on as soon as it arrives
  muleFlag: false,        // your bank's mule check flagged the payee
  screen: false, call: false, pressure: false,
};

export function featuresFrom(a) {
  const amount = Math.max(1, Number(a.amount)), usual = Math.max(1, Number(a.usual));
  const r = amount / usual;
  const earlier = Math.max(0, Math.round(Number(a.earlierToday) || 0));
  const v = (a.firstTime ? 0 : REGULAR_V) + earlier;
  const pair = (a.firstTime ? 0 : REGULAR_PAIR) + earlier * r;
  const probes = Math.max(0, Math.round(Number(a.probes) || 0));
  const snd = { n_t: a.manyNew ? 75 : 53, e_t: a.firstTime ? r : 0.01, g_t: 1, z_t: Math.log(r),
                pi_t: probes, d_R: 3, v_t: v, zc_t: Math.log(r + pair), ptr_t: 0.57 };
  const rcv = { n_t: 50, e_t: 0.01, g_t: 1, z_t: 0.06, pi_t: probes, d_R: Number(a.link),
                v_t: v, zc_t: Math.log(1.06 + pair), ptr_t: a.forwards ? 0.95 : 0.43 };
  const larger = (x, y) => (Math.abs(x) >= Math.abs(y) ? x : y);
  const comb = {
    n_t: Math.max(snd.n_t, rcv.n_t), e_t: Math.max(snd.e_t, rcv.e_t), g_t: 1,
    z_t: larger(snd.z_t, rcv.z_t), pi_t: Math.max(snd.pi_t, rcv.pi_t), d_R: Math.min(snd.d_R, rcv.d_R),
    v_t: Math.max(snd.v_t, rcv.v_t), zc_t: larger(snd.zc_t, rcv.zc_t), ptr_t: Math.max(snd.ptr_t, rcv.ptr_t),
  };
  const f = {};
  for (const b of BASES) { f[b] = comb[b]; f[`${b}_snd`] = snd[b]; f[`${b}_rcv`] = rcv[b]; }
  return f;
}

/** A scenario-shaped object the app can load and pay like the built-in ones. */
export function customScenario(a) {
  const name = String(a.payee || "").trim() || "Someone";
  const note = String(a.note || "").trim();
  return {
    key: "custom",
    custom: true,
    label: { en: "Your own payment", hi: "आपका अपना भुगतान" },
    payee: { en: name, hi: name },
    upi: String(a.upi || "").trim() || "unknown@upi",
    note: { en: note, hi: note },
    amount_inr: Math.max(1, Number(a.amount)),
    features: featuresFrom(a),
    soft_signals: { is_screen_shared: Boolean(a.screen), is_voice_active: Boolean(a.call),
                    chat_stage: a.pressure ? "payment_demand" : "none" },
    mulehunter_status: a.muleFlag ? "mule" : "clean",
  };
}
