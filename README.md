---
title: Maya
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Stopping UPI scams at the moment of payment (demo)
---

# Maya — UPI scam check demo

Open **`/demo/`** on the big screen and scan its QR code with a phone (or open
**`/app/`** and enter the code). Pick a payment and tap Pay: Maya decides
release / nudge / hold, and the demo screen shows why.

- Everything is **synthetic**: no real payments, bank data or accounts.
  MuleHunter, DoT FRI, 1930 and the chat/call signals are simulated.
- The model was trained on generated data; its near-perfect scores say nothing
  about real-world accuracy.
- Demo rooms live in memory and reset when the Space restarts or sleeps.
- API reference: `/docs`.

Built from the `maya-fraud-detection` repository with `scripts/build_space.py`.
