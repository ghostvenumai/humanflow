# HumanFlow Challenge Preflight Report

- Script version: 1.1.0
- Generated: 2026-08-23T16:49:22+02:00
- Host: serverserver-ThinkPad-T470-W10DG
- Project directory: /home/serverserver/HumanFlow-PreStart/humanflow
- Mode: full
- Fable test requested: 0
- Audio test requested: 1

| Status | Area | Check | Detail |
|---|---|---|---|

## 1. Master Briefing

| PASS | Briefing | PDF readable | /home/serverserver/HumanFlow-PreStart/HumanFlow_Master_Briefing_V1_1.pdf |
| INFO | Briefing | SHA-256 | 05d49a32191dd8fa257077733e5194d800a1a708acaee41398485eca4b127237 |
| PASS | Briefing | Page count | 37 pages |
| PASS | Briefing | Section present | HumanFlow Autonomous Quality Loop |
| PASS | Briefing | Section present | Codex integration |
| PASS | Briefing | Section present | Claude Code integration and Fable 5 |
| PASS | Briefing | Section present | Dynamic Model Router |
| PASS | Briefing | Section present | Coding-Agent Tournament Mode |
| PASS | Briefing | Section present | HumanFlow Scorecard System |
| PASS | Briefing | Section present | Controlled Continuous Improvement |
| PASS | Briefing | Section present | SAFE-DEMO mode |
| WARN | Briefing | Version labeling | V1.1 extension exists, while cover/header still says V1. Clean this before final submission. |
| WARN | Briefing | Contents may be stale | The V1.1 Scorecard/Continuous Improvement extension was appended after the original contents. |

## 2. Host / Time / Capacity

| INFO | Host | Operating system | Ubuntu 24.04.4 LTS |
| INFO | Host | Kernel / architecture | Linux 6.8.0-138-generic x86_64 GNU/Linux |
| PASS | Host | CPU threads | 4 |
| PASS | Host | RAM | ~15 GiB |
| PASS | Host | Free disk | ~136 GiB |
| INFO | Time | Timezone | Europe/Berlin |
| PASS | Time | Clock synchronized | NTP synchronized |
| MANUAL | Challenge | Three free days | Choose the start date only after all technical checks are green. Use --start-date YYYY-MM-DD to calculate the 72-hour window. |

## 3. Development Toolchain

| PASS | Toolchain | git | git version 2.43.0 |
| PASS | Toolchain | curl | curl 8.5.0 (x86_64-pc-linux-gnu) libcurl/8.5.0 OpenSSL/3.0.13 zlib/1.3 brotli/1.1.0 zstd/1.5.5 libidn2/2.3.7 libpsl/0.21.2 (+libidn2/2.3.7) libssh/0.10.6/openssl/zlib nghttp2/1.59.0 librtmp/2.3 OpenLDAP/2.6.10 |
| PASS | Toolchain | python3 | Python 3.12.3 |
| PASS | Toolchain | node | v18.19.1 |
| PASS | Toolchain | npm | npm 9.2.0 |
| PASS | Toolchain | make | GNU Make 4.3 |
| PASS | Toolchain | jq | jq-1.7 |
| PASS | Toolchain | ffmpeg | ffmpeg version 6.1.1-3ubuntu5+esm10 Copyright (c) 2000-2023 the FFmpeg developers |
| PASS | Toolchain | ffprobe | ffprobe version 6.1.1-3ubuntu5+esm10 Copyright (c) 2007-2023 the FFmpeg developers |
| PASS | Toolchain | gh | gh version 2.45.0 (2026-03-17 Ubuntu 2.45.0-1ubuntu0.3+esm3) |
| WARN | Toolchain | shellcheck missing | Recommended; not a hard blocker for initial coding. |
| PASS | Browser | Browser available | google-chrome |

## 4. Git / GitHub / Tournament Isolation

| PASS | Git | user.name configured | Serkan Iazurlo |
| PASS | Git | user.email configured | ghostvenumai@proton.me |
| WARN | Git | Project directory does not exist | /home/serverserver/HumanFlow-PreStart/humanflow (do not create challenge code early if rules forbid it) |
| PASS | GitHub | GitHub CLI authenticated | gh auth status succeeded |
| PASS | Tournament | Git worktrees | Local isolated candidate worktree creation works. |

## 5. OpenAI Codex

| PASS | Codex | Unprivileged user namespaces | kernel.unprivileged_userns_clone=1 |
| PASS | Codex | User namespace capacity | user.max_user_namespaces=62766 |
| WARN | Codex | User-namespace smoke test | unshare -Ur true failed. If Codex shows a bubblewrap/user-namespace warning, fix this before challenge start. |
| PASS | Codex | CLI installed | codex-cli 0.149.0 |
| PASS | Codex | exec supports --sandbox | Required/valuable for HumanFlow automation. |
| PASS | Codex | exec supports --json | Required/valuable for HumanFlow automation. |
| PASS | Codex | exec supports --ephemeral | Required/valuable for HumanFlow automation. |
| PASS | Codex | exec supports --model | Required/valuable for HumanFlow automation. |
| PASS | Codex | codex doctor | Installation/config/auth/runtime/Git diagnostics completed. |
| PASS | Codex | Model catalog query | codex debug models succeeded. |
| PASS | Codex | Model visible | gpt-5.6-luna |
| PASS | Codex | Model visible | gpt-5.6-terra |
| PASS | Codex | Model visible | gpt-5.6-sol |
| PASS | Codex | Live non-interactive smoke test | codex exec completed successfully. |
| PASS | Codex | workspace-write sandbox smoke test | Codex successfully wrote the expected file inside a disposable temp workspace. |

## 6. Claude Code / Fable 5

| PASS | Claude | CLI installed | 2.1.229 (Claude Code) |
| PASS | Fable | CLI version supports Fable selection | 2.1.229 >= 2.1.170 |
| PASS | Claude | Budget-cap enforcement version | 2.1.229 >= 2.1.217 |
| PASS | Claude | Authentication | claude auth status succeeded. |
| PASS | Claude | CLI supports --model | Detected in claude --help. |
| PASS | Claude | CLI supports --max-budget-usd | Detected in claude --help. |
| PASS | Claude | CLI supports --output-format | Detected in claude --help. |
| PASS | Claude | CLI supports --permission-mode | Detected in claude --help. |
| PASS | Claude | CLI supports --tools | Detected in claude --help. |
| PASS | Claude | Live Sonnet smoke test | Non-interactive Claude Code request succeeded. |
| MANUAL | Fable | Fable 5 organization access | Not proven without a real request. Run with --fable-test only when you accept a tiny usage-credit charge. |

## 7. Docker

| PASS | Docker | CLI installed | Docker version 29.1.3, build 29.1.3-0ubuntu3~24.04.2 |
| PASS | Docker | Daemon reachable | docker info succeeded. |
| PASS | Docker | Compose plugin | Docker Compose version 2.40.3+ds1-0ubuntu1~24.04.1 |
| PASS | Docker | User in docker group | No sudo normally required. |
| PASS | Docker | Container smoke test | hello-world ran successfully. |

## 8. Audio / Microphone / WebRTC Readiness

| PASS | Audio | PulseAudio/PipeWire control | PulseAudio (on PipeWire 1.0.5) |
| PASS | Audio | Default input source | alsa_input.pci-0000_00_1f.3.analog-stereo |
| PASS | Audio | Input sources detected | 1 non-monitor source(s) |
| PASS | Audio | 2-second microphone capture | Recorded 64044 bytes. duration=2.000000s |
| MANUAL | WebRTC | Browser microphone permission | Open a localhost HTTPS/allowed-origin demo and confirm microphone permission + playback once HumanFlow demo exists. |

## 9. Credentials / API Readiness

| INFO | Secrets | OPENAI_API_KEY not set | Optional until this provider is selected. |
| PASS | Secrets | ANTHROPIC_API_KEY present | Value intentionally hidden. |
| INFO | Secrets | DEEPGRAM_API_KEY not set | Optional until this provider is selected. |
| INFO | Secrets | CARTESIA_API_KEY not set | Optional until this provider is selected. |
| INFO | Secrets | LIVEKIT_URL not set | Optional until this provider is selected. |
| INFO | Secrets | LIVEKIT_API_KEY not set | Optional until this provider is selected. |
| INFO | Secrets | LIVEKIT_API_SECRET not set | Optional until this provider is selected. |
| INFO | Secrets | TWILIO_ACCOUNT_SID not set | Optional until this provider is selected. |
| INFO | Secrets | TWILIO_AUTH_TOKEN not set | Optional until this provider is selected. |

## 10. Claude/Fable Budget Guard

| MANUAL | Budget | Budget config not built yet | Expected before optimization loop: config/development-budget.yaml and/or config/model-router.yaml. |
| MANUAL | Budget | Actual Anthropic credit balance | A local script cannot reliably prove your Anthropic Console/usage-credit balance. Confirm >=100 USD in the billing UI before start. |

## 11. Network / Challenge Page

| PASS | Network | DNS: github.com | Resolved. |
| PASS | Network | DNS: api.openai.com | Resolved. |
| PASS | Network | DNS: api.anthropic.com | Resolved. |
| PASS | Challenge | Public page reachable | GET only; no form submitted. |
| PASS | Challenge | Expected challenge wording found | Page contains one or more known terms (3 Tage / Briefing / Demo-Call / GitHub). |
| INFO | Challenge | Page snapshot SHA-256 | bd1bedc755017dbc00a70451c1f83fb5daff0324d8c2724ec72c8a8ad7874ba1 |

## 12. HumanFlow Project Artifacts


## 13. Quality Loop Preconditions


## 14. Manual Go/No-Go Items

| MANUAL | Challenge | Official start not triggered | Confirm you have NOT selected/triggered the official start before the machine is ready. |
| MANUAL | Challenge | Official start email/briefing | At start, upload the official Everlast task + scoring criteria before implementing challenge-specific features. |
| MANUAL | Calendar | 72 uninterrupted hours | Reserve the three-day window plus submission buffer. |
| MANUAL | Billing | Anthropic >=100 USD available | Confirm in Anthropic Console/usage-credit UI. |
| MANUAL | Audio | Headset/mic real conversation test | Perform one browser call with headphones and one with speakers to expose echo/double-talk behavior. |
| MANUAL | GitHub | Submission visibility/permissions | Confirm repo visibility and access match the official Everlast briefing once received. |

## Summary


## Final Summary

- PASS: 68
- WARN: 5
- FAIL: 0
- MANUAL: 11
- Automated readiness (PASS / automated checks): 93%

### Go/No-Go rule

Do **not** start the official 72-hour challenge while any critical FAIL remains in:
Codex, Claude authentication, Docker daemon/Compose, Git worktrees, microphone capture
(after running '--audio-test'), or system clock synchronization.

The following cannot be fully proven by this local checker and remain manual:
actual Anthropic account balance, official Everlast rules received on start day,
and whether the chosen three-day window is genuinely free.

PREFLIGHT_COMPLETE=1
