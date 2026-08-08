# System architecture — Pi / G5 role split

## Machines

| | Pi 3 B (`dicecam`, 10.0.0.23) | GMKtec G5 (`NucBoxG5`, 10.0.0.81) | **TheBeast** (10.0.0.5) |
|---|---|---|---|
| CPU | BCM2837 4×A53 @1.2GHz | Intel N97 (4×Alder Lake-N) | i9-13900K |
| RAM | 905 MiB usable | 12 GB | 63.8 GB |
| OS | RPi OS Trixie, 64-bit | Windows 11, PS 5.1, Python 3.12 | Windows 11, Python 3.14 |
| GPU | VideoCore IV | UHD iGPU, **no discrete GPU** | **RTX 4090** |
| Role | sensor head | **unassigned — see below** | control panel + model inference |

### 10.0.0.5 is TheBeast — the workstation itself

Not a separate server: it is the machine this project is developed on. That is
why the control panel needs no SSH deployment, and it explains 42 tok/s on a
27B model. It runs Ollama 0.32.6 with **`gemma3:27b` (vision-capable)**, plus
`qwen2.5:14b`, `llama3.1:8b`, `mistral`, `bge-m3` and a custom
`campaign-dm:latest`.

## Claude Code CAN read images — as files

swadeledger marks `claude-code` as unable to see pictures, and for its workload
that is right: it passes page data inline and `claude -p` has no flag for an
image. **That limitation does not apply here.** Our image is a file on disk, and
Claude Code can open a file with Read.

Measured on the 8-dice tray (ground truth 4x d20, 4x d12):

| provider | type accuracy | time |
|---|---|---|
| Ollama gemma3:27b | 4/4 d20, **0/4 d12** (all called d10) | ~3 s |
| Claude Code + haiku | 6 d20 / 2 d12 — **2 wrong** | 10 s |
| **Claude Code + opus** | **4 d20 + 4 d12 — exactly right** | 105 s |

Opus also matched an independent human reading on 6-7 of 8 values. It is the
only configuration that has solved the d12/d20 confusion, and under a
subscription it costs nothing per roll.

The cost is latency: `claude -p` is an agent, not a completion call. Unbounded
it took 113 s; with swadeledger's constraining flags, 79 s; on a full tray,
~105 s and 7 476 output tokens. So the ensemble defaults to **one prompt** for
this provider rather than three.

That trade was originally acceptable because the Pi's independent count still
caught a missed die. That count is gone — it reported up to 73 dice for 8 — so
per-die disagreement across prompts is now the *only* confidence signal, and
dropping to one prompt gives it up entirely. Batching makes the second prompt
affordable again (~3x faster than per-die calls), which is the reason to keep
paying for it.

### Flags, and the one deliberate difference

swadeledger strips every tool including Read, because raw PDF text lands in the
instruction position of an agent holding credentials. **We cannot strip Read** —
it is how the agent sees the picture at all. So the blast radius is contained
differently:

- working directory is a fresh temp dir containing **only** the tray image
- `--allowedTools Read`, everything else refused by name
- `--strict-mcp-config --mcp-config '{"mcpServers":{}}'` so nothing declared
  elsewhere can attach
- `--json-schema` for structured output, `--output-format json` for the envelope
- subprocess timeout as the backstop

A photograph is a far narrower injection surface than arbitrary book text, but
it is not zero — someone could photograph written instructions — and Read is
scoped accordingly.

### Two credentials that are trivially confusable

`sk-ant-oat...` is a Claude Code OAuth token (subprocess). `sk-ant-api...` is an
API key (`x-api-key` on the Messages API). Both come from Anthropic, both start
`sk-ant-`, and pasting the OAuth token into the API-key field returns
**`401 invalid x-api-key`** — which reads like a bad key rather than the right
credential behind the wrong door. That happened here.

So credentials are now filed by **what they are**, not by which box they were
typed into: `settings.classify()` routes on prefix, the UI says where it put it
and why, and a one-time migration moved the already-misfiled token.

## Anthropic API path, with Ollama as fallback

Modelled on the user's own `swadeledger` project (`pocketbase/pb_hooks/ai_lib.js`),
which solves the same problem and had already worked out the sharp edges.

**Claude Code cannot do this job**, and that is not a guess — swadeledger states
it outright and enforces it:

```js
const CAN_SEE = { anthropic: true, ollama: true, "claude-code": false };
```
> *"`claude -p` takes a prompt on the command line and has no flag for an image."*

Its `chosenProvider()` drops the named engine whenever a request carries images
and routes to Anthropic or Ollama instead — added after the absence of that
check silently lost ten pages of a rulebook. Dice reading is entirely images, so
following what that project actually does means using the Anthropic path here.
The `CLAUDE_CODE_OAUTH_TOKEN` remains the right credential for the *text* work
it does there.

Patterns copied deliberately:

- **Image block before the text.** Documented ordering, and the honest one —
  everything the instructions refer to is in the picture.
- **`output_config` with a `json_schema`.** The reply shape is enforced at the
  API instead of parsed out of prose, which deletes the whole class of "wrote a
  sentence instead of JSON" failures. The Ollama path still needs the lenient
  parser; the Anthropic path does not.
- **`cache_control: ephemeral` on the system block** — identical for every
  roll, so pay for it once.
- **Explicit `refusal` and `max_tokens` handling.** A classifier refusal
  arrives as an ordinary 200 with empty content; reading `content[0]` blindly
  turns that into a confusing crash.
- **`effort` for non-haiku models only.**
- **Key read from the environment, used at exactly one call site**, never
  logged or returned.

Added here: images are capped at **1568 px on the long edge** before sending.
A full-sensor tray crop is 1980x1900, and above that cap the API downscales
anyway — so sending more is paying to transmit detail that gets discarded.

Selection is `auto` (Anthropic when `ANTHROPIC_API_KEY` is set, else Ollama),
overridable per roll from the Runtime tab. Keeping Ollama as fallback matters
for something that has to work at a table: it costs nothing and survives the
network being down.

## Camera runs at full sensor resolution

`MAIN_SIZE` is 3280x2464. Measured justification: at 1640x1232 the reader typed
d20s unstably; at full resolution it got **4/4** d20s right and the residual
error became a consistent d12->d10 confusion rather than noise. Silhouette
circularity was unchanged, so this buys nothing for the CV path — it buys
accuracy in the model, which is where reading happens.

Costs on a Pi 3: ~24 MB per RGB888 frame against 905 MB usable (measured 617 MB
in use, stable), single-digit frame rate on the main stream, and `/roll` rising
from ~3 s to ~15 s. Acceptable because nothing needs a fast main stream — the
MJPEG preview comes from the untouched 640x480 lores stream.

### Where dice reading runs — decided 2026-08-06

**Remotely.** Either the Ollama host or a hosted AI subscription. It will *not*
run on the Pi or the G5.

This settles what was an open fork, and it simplifies things:

- No need to train a small classifier just to fit the G5's N97. Approaches 1
  and 2 in `dice-reading.md` lose their main justification and drop in
  priority; approach 3 is the plan.
- The Pi's job is unchanged and remains the important one: capture, settle
  detection, tray masking, detection, cropping. **Crop quality is now the whole
  ballgame** — a remote reader can only be as good as the crop it is sent, and
  network round-trips make it expensive to send anything but a good one.
- Latency budget shifts. ~1 s/die local on the 4090 is a floor, not a ceiling;
  a subscription API adds round-trip and rate limits, so batching crops into
  one request matters more than shaving inference time.
- It introduces a dependency the table cannot control — network, and possibly a
  paid service. Worth an offline fallback eventually, but not now.

The G5 remains unassigned. It is genuinely not needed for inference under this
plan.

## Division of work

**Pi does capture and reduction. G5 does inference.**

The Pi is a poor inference host — four A53 cores and under a gigabyte of RAM —
but it is the only machine physically attached to the camera, and it is
perfectly capable of the cheap, high-volume work:

- camera capture at locked exposure/gain/white balance
- settle detection (frame differencing) so we only process a stopped tray
- tray masking + local-variance dice detection
- cropping, and shipping crops onward

That reduction matters: a full frame is 3280×2464, while seven die crops are a
few hundred KB. Sending crops rather than frames cuts network volume by well
over an order of magnitude and keeps the classifier's input canonical.

The G5 takes crops and returns values.

### Caveat on the G5

The N97 has **no discrete GPU**. It is fine for *inference* on a small
classifier, but it is a poor training box. Plan to train elsewhere (or accept
slow CPU training on a deliberately small model) and deploy the trained weights
to the G5. Do not design around the assumption that the G5 can iterate on
training quickly.

## Phases

These have different topologies and conflating them wastes effort.

### Prototype / data-collection phase (now)

Offline and batch. No live link needed.

```
Pi: capture -> detect -> crops on local disk
     |
     +-- pulled in bulk (scp) --> G5 / workstation
                                    |
                                    +-- label, train, and compare the three
                                        stage-2 approaches on identical held-out frames
```

The three approaches under comparison (see `dice-reading.md`): whole-crop
classifier, geometric top-face, and a vision model. They share one crop
pipeline so the comparison is apples-to-apples — the crop pipeline is
therefore the piece to get right first.

### Runtime phase (later)

```
Pi: capture -> settle-detect -> detect -> crops --HTTP POST--> G5: classify -> values
```

Transport candidates, in order of preference:

1. **HTTP POST to a small service on the G5** — simplest to debug, easy to
   version, natural request/response for "here are 7 crops, give me 7 values".
2. **SMB share** — port 445 is already open on the G5, so it needs no new
   service, but it is a file-drop with no natural response channel.
3. **MQTT** — worth it only if more devices join later.

Start with (1).

## Connectivity status

| link | state |
|---|---|
| workstation → Pi | working, key auth, user `paul` |
| workstation → G5 | **RDP 3389 and SMB 445 open; SSH closed** — needs OpenSSH Server enabled |
| Pi → G5 | not yet established |

### Enabling SSH on the G5

Run as Administrator on the G5:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

**The gotcha:** if the account is an Administrator, Windows OpenSSH ignores
`~/.ssh/authorized_keys` and reads `C:\ProgramData\ssh\administrators_authorized_keys`
instead, which must also have inheritance stripped. Putting the key in the
usual place and getting "Permission denied" with no explanation is the normal
first experience here.

```powershell
$key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrZdC9Ue99NLFY6WalXt6FqGK9p30V2oiuDYwHEiFRh paul@TheBeast'
$f = 'C:\ProgramData\ssh\administrators_authorized_keys'
Add-Content -Path $f -Value $key
icacls $f /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F'
```

Optional, but makes remote work much less painful — default to PowerShell
rather than cmd:

```powershell
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
  -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
  -PropertyType String -Force
```
