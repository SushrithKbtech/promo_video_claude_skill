---
name: promo-video
description: Make a broadcast-quality promo video (MP4, 16:9 / 9:16 / 1:1) for any website, app or product, rendered entirely in code around one continuous light - six film formats (journey, keynote, launch, story, showcase, reel) so every product gets a different shape, real or rebuilt UI, kinetic type, device loops, product shots, a voice-over, an original per-product score and full sound design. Use when the user asks for a promo, product video, launch film, ad, spec ad, demo video, teaser, reel or brand animation for a URL, an app, a product or an idea, in a light or dark look.
---

# promo-video

You are the director, editor, motion designer and sound designer of a short product
film. Everything is drawn by code (Python + ffmpeg). No AI video, no stock footage.

The user gives you as little as a name or URL ("make one for linear.app") or as much as a
full brief (script, length, 9:16, dark, "Apple style" music, their own track, their
screenshots). Fill every gap yourself with taste: do not interview them with a list of questions.
Instead, research the product, design a complete film, and propose it as a short storyboard they
can approve with one click or change (Step 2). Only if you cannot tell what the product is, ask
that first.

---

## 0. Set up (every run, takes seconds)

`SKILL_DIR` is the folder that contains this SKILL.md. The engine (`run.py`, `adkit/`)
normally sits next to it. If someone installed only this file, fetch the engine once:

```bash
SKILL_DIR="<folder containing this SKILL.md>"
ENGINE="$SKILL_DIR"
if [ ! -f "$ENGINE/run.py" ]; then
  ENGINE="$SKILL_DIR/.engine"
  [ -f "$ENGINE/run.py" ] || git clone --depth 1 https://github.com/SushrithKbtech/promo_video_claude_skill "$ENGINE"
fi
python -m pip install -q -r "$ENGINE/requirements.txt"
python "$ENGINE/run.py" doctor
```

Use `python` on Windows and `python3` elsewhere if `python` is missing. `doctor` checks
numpy, OpenCV, Pillow, scipy, ffmpeg, a headless Chrome/Edge (for SVG logos and website
capture) and the free neural TTS. ffmpeg must be on PATH; tell the user how to install it
if it is missing (`winget install ffmpeg`, `brew install ffmpeg`, `apt install ffmpeg`).

Every command below is `python "$ENGINE/run.py" <command> <PROJECT>`. A project is a
folder (put it where the user works, e.g. `./promo-<product>/`) holding `brief.json`,
`assets/`, and the outputs in `build/` and `out/`.

---

## 1. The method (what makes these films look expensive)

1. **One continuous light runs the whole film and never stops moving.** It enters on frame 0,
   becomes the logo, leaves it, does every job in the product, races back, and ends as the line
   on the end card. Nothing pops in from nowhere.
2. **Find the idea already inside the logo.** A logo with N colour facets gives N lights. Each
   light does ONE job (save, find, share / design, prototype, ship / plan, track, pay). On the
   logo hit they slot back into their facets.
3. **On a light stage, light is ink.** Additive glow vanishes on white, so the engine composites
   light as coloured ink. On a dark stage it is glow. Never draw white cores or flashes on white.
4. **Curves, not spirals.** Lights travel soft Beziers with fading tails (intensity w^2).
   Rigid circles look fake.
5. **The hook is one slam on the first downbeat.** Readable in a fraction of a second; the logo is
   the first thing anyone sees.
6. **Record the voice first, time the KEY WORD, not the line.** Each line is placed so its key word
   lands on its picture beat. Never time-stretch a voice.
7. **Put the logo on the song's drop.** The logo hit is a downbeat; the score (or the user's track)
   is built or offset so its biggest moment lands there.
8. **Every visible move has a sound.** ~50 cues: whooshes, impacts, ticks, keys, pops, chimes.
8b. **The picture moves with the music.** Plain is the enemy. In the song's loud sections the
   whole frame kicks on every beat (harder on downbeats); over the bar before each drop the camera
   sucks in, the screen empties on the last beat, and the drop lands with a flash in the brand
   colour, an RGB split, a shockwave ring of light, camera shake and a sub impact. Hard cuts in
   loud sections get a split too. The section energy comes from the song itself (its low end per
   bar), so quiet intros stay quiet and drops explode. `"energy": "hype" | "normal" | "calm"`
   sets how hard (default from the format: launch and reel hype, showcase calm, others normal).
   Use "hype" whenever the user asks for energy, a launch, social, or "keep people watching".
9. **Iterate cheap, render once.** Half-res stills and a contact sheet first. Look at them.

---

## 2. Workflow

### Step 1: understand the product (2-5 minutes, no questions)

- **URL given:** `python "$ENGINE/run.py" capture <url> <PROJECT>` writes `capture.json` (title,
  description, headings, CTAs, nav, CSS brand colours, fonts, Google Fonts) plus
  `assets/screens/desktop.png`, `mobile.png` and `assets/logo_candidates/`. It also scrolls the
  whole page in a headless Chrome so sections that animate in on scroll are captured
  (`assets/screens/scroll/home_00.png ...`, 2x). `--pages about,pricing` scrolls more pages;
  `--hold 765:6:3.5` takes 6 shots 3.5 s apart at scroll position 765 px (for a live demo, a chat
  or a ticker that plays on the page: the later shots show it finished). It only scrolls and
  looks. Read capture.json, Read the screenshots and the logo candidates (images) and decide what
  the product does. If capture fails (bot protection), use WebFetch / the browser tools or what
  you know.
- **App / product / idea without a site:** use what the user wrote. Invent nothing that sounds like
  a claim (no fake awards, prices, user counts); keep demo data obviously illustrative.
- **Logo:** prefer an official SVG (from logo_candidates, the site, or the user). PNG works too.
  Copy it to `assets/logo.svg|png`. Never redraw, recolour or distort it. If there is truly no
  logo, make a simple wordmark-only film: set `brand.logo` to a clean monogram SVG you write
  (a rounded square in the primary colour with the initial) and say so.
- **Font:** the brand's font if it is on Google Fonts (`font.family`), else Inter.
- **Real UI beats rebuilt UI.** When the user wants their actual product shown (or it is their
  own product), get real screens and use the `screen` scene (and `devices` / `phone` with
  real screenshots):
  - screenshots the user gives you, or `capture` output (`assets/screens/`);
  - pages of the live site: `capture` each URL, e.g. `/pricing`, `/dashboard` if public;
  - a walkthrough / demo video in the repo or site: pull frames with
    `ffmpeg -ss <t> -i demo.mp4 -frames:v 1 shot.png`, make a contact sheet first
    (`-vf fps=1/3,scale=320:-1,tile=6x7`) to find the best moments, then pass `crop` to cut the
    UI out of the frame and measure highlight rects in the frame's pixel coordinates.
  Never type into, sign up on, or submit forms on someone else's live site to get a screen; ask
  the user for screenshots instead. Fill only fictional data when a screen needs content.

### Step 2: direct the film, then PROPOSE it and ask

Decide, in this order:

0. **Film format:** the shape of the film (`journey`, `keynote`, `launch`, `story`, `showcase`,
   `reel`; section 3). Pick from the product and the user's words, and not the one used last time.
0b. **Signature:** the shape the light takes, from what the product *is* (section 3, "Signature").
   A health product's light is a heartbeat; an audio product's is a sound wave. Plan one hero
   moment with the `signature` scene when it fits the story.
1. **Size and length:** duration (default 21.8 s; 15 s for social cut-downs, 30 s for launches),
   aspect (16:9 default; 9:16 for Reels/TikTok/Shorts, captions turn on automatically), 60 fps.
2. **Look:** `stage.mode` light (clean SaaS, productivity, consumer) or dark (dev tools, AI,
   fintech, gaming, automotive, luxury, night brands). Match the product's own site.
3. **Lights:** default = the logo's facet colours. Monochrome logo: pick 3 colours from the site
   (`lights`), or one brand colour plus two neighbours.
4. **The jobs:** turn the product into 3-5 verbs, one scene each (table below). Mix scene types;
   never the same scene twice in a row. End on `devices` when the product is multi-platform,
   otherwise on the strongest proof (`chart`, `features`, `product`).
5. **Transitions:** the light hand-over (the light flies from one scene into the next in the
   signature's shape and changes colour along its tail), `whip` (fast pan with motion blur),
   `cut` (hard cut on a beat), `wipe` (a line of light sweeps across and reveals the next scene),
   `push` (the camera flies through the outgoing scene into the next), `shape` (the next scene is
   revealed inside a growing, turning shape edged with light: act `"shape": "star" | "circle" |
   "logo"`; the logo kind opens through the brand's own mark). Containers persist
   automatically: search -> share (a row grows into the page), share -> devices (the page flies
   into the laptop). Mix at least three kinds in a film; never the same one three times running.
6. **Hook:** `"hook": "converge"` (lights sweep in from the edges into the facets) or `"trace"`
   (one light draws the logo's silhouette, then the fill slams in; best for single-shape marks).
7. **Script:** write it like a copywriter, not a feature list.
   - Arc: the human tension (what is hard today) -> the turn (the product) -> 2-4 proof beats
     (what you do with it, in verbs) -> the brand name on the hit. A cold open starts on the
     tension line; a slam opening starts on the proof.
   - One line per beat, 3-8 words, present tense, second person ("Describe any situation in a
     sentence."). One key word per line, chosen to land on the picture beat.
   - Rhythm: parallel structure and triads ("Every file. Every photo. Every idea."), then a short
     line that breaks the pattern ("One place."). Read it aloud; cut every word that does not earn it.
   - Banned: "revolutionary", "seamless", "unlock", "leverage", "game-changer", "all-in-one",
     invented numbers, claims the product does not make. Use the product's own words where good.
   - The last line is the brand name after the hit (`"at": "hit+0.3", "role": "brand"`); the end
     card carries the tagline or the URL, not the same sentence as the VO.
8. **Sound:** music mood + voice (section 5).

| product does... | scene | the light becomes |
|---|---|---|
| store, import, back up, sync files | `upload` | the progress bar |
| search, find, retrieve, AI answers | `search` | the focus line, then traces the result |
| collaborate, share, comment, multiplayer | `share` | waits on Share, splits into live cursors |
| automate, integrate, pipelines, prototypes | `flow` | runs node to node, leaving lit connectors |
| analytics, growth, finance, health metrics | `chart` | draws the line; the number rolls |
| AI assistant, chat, copilots | `chat` | the streaming caret of the answer |
| a real UI you want to show | `screen` | traces highlights on the real screenshot |
| a manifesto line, positioning | `statement` | writes the words, circles the key one |
| many features at once | `features` | traces each bento card as it pops |
| mobile apps | `phone` | the tap, then the notification outline |
| physical products (car, device, bottle) | `product` | rim light on the silhouette, spec leaders |
| works everywhere | `devices` | three lights chase a loop behind the devices |
| the product's own essence (hero moment) | `signature` | the product's motif drawn across the frame |
| the UI as a physical object, cinematic | `diorama` | lanes and a playhead on a 3D set; a mascot hops across it |
| a bold claim, a quote, a number | `card` | underlines the headline, the stat's line draws as it counts |
| a line said with weight, words on the beat | `type` | six kinetic-type styles (below) |
| one thing becomes the result (ask -> answer) | `morph` | outlines the container as it grows pill -> bar -> card |
| many things come together, a team, inputs | `orbit` | rings the product as everything converges into it |
| breadth: a library, many templates or cards | `tunnel` | hairlines on cards circling a 3D ring around the line |
| integrations, a bench of specialists, sources | `grid` | tiles pop onto a tilted 3D plane, one per beat |
| a finished task, a win, "you're done" | `success` | draws the check; 3D confetti bursts |
| energy, real-world footage, launch teasers | `montage` | cuts the user's images on every beat with a push-in |

`type` styles (`data.style`), each landing on the song's beats, and on the narration's own words
when the voice-over says the same line (each word appears as it is spoken):
- `build`: words land one by one around an inline icon (`icon`: "logo" | "star"), the line
  re-centring as it grows. A product name or a promise.
- `swap`: a sentence rewrites itself (`phrases`), shared words stay put, new ones rise in. A
  problem framed in three beats ("Confident." "Confident. Fluent." "Confident. Fluent. Unchecked.").
- `giant`: one huge gradient word (`word`, `from`, `to`) that springs in, then shrinks for a `sub`.
- `cloud`: keywords drift at depth around a focus word that changes on the beat (`focus`, `words`).
- `tilt`: a phrase receding in 3D perspective (`text`). Calm, playful.
- `teaser`: wide-tracked letters blur in, then slide out sideways (`text`). Pre-drop and "coming soon".
Any style: `"font": "display"` sets it in the brand's display face (`font.display`), `highlight`
colours words in the scene's light, `size` sets the size.

`montage` needs `images` (product shots, frames from a demo video, real UI captures) and takes
`every` (beats per shot) and `grade` ("teal-orange" for footage, "none" for UI). `tunnel` takes
`images` (cards; real UI cards cropped from captures are best), `words` and `text`. `orbit` takes
`items` ([{label, color}] 4-8), `text`, `converge`. `grid` takes `items` and `title`. `morph`
takes `steps` ([{w, h, r, label, icon, glow, lines}]) and `cursor`. `success` takes `text`, `sub`.

`diorama` is the most cinematic scene: the real (or a generated) UI lies on a table like a
physical slab and a 3D camera flies over it (grazing angles, macro depth of field, warm light,
contact shadow). Use it once per film, for the hero moment, ideally on the product's own UI. A
cube mascot in the brand colour can hop across it (squash and stretch, landing debris, a glowing
playhead); drop the mascot for serious brands. `card` gives a premium social-carousel rhythm
(condensed display type, highlight words in the light colour) between product shots; never more
than two cards in a row.

The order of scenes is a story, not a checklist: start where the viewer's problem is (cold open) or
where the brand is (slam), build to the most impressive proof, and land the logo on the music's
biggest moment. Two products should never share the same scene order.

**Propose the film and ask before building.** Show the plan as a storyboard the user can read in
ten seconds, then ask. Example:

```
FIGMA · "launch" format · 21.8 s · 16:9 · light stage · signature: spark
Music: hype (its own song) · SFX palette: punchy · Voice: Ava (female, warm)

0.0   Trace hook: one light draws the Figma mark -> SLAM on the downbeat
1.1   Light writes "Figma"
3.0   share     Live cursors fly onto a design canvas   "Design together."
8.6   flow      (whip) light runs Home > Cart > Checkout  "Prototype in minutes."
11.0  chat      (wipe) a prompt becomes a pricing page    "Just ask."
13.9  devices   (push) the page lands on laptop, tablet, phone  "On every screen."
16.3  hard CUT to the logo on the drop -> "Figma." -> end card "made by ..."
```

Then ask, with the AskUserQuestion tool when it is available (otherwise a plain question), for
example: "Render this?" with options "Yes, make it" / "Change the script" / "Different style or
music" / "Other".

**Always ask where the music comes from**, in the same AskUserQuestion call as a second question,
unless the user already gave a track or said which: "Music: Claude's own score, or your track?"
with options like "Own score: <the mood you picked> (Recommended)" / "Own score, different mood" /
"I'll give you a track (mp3/wav path)". With a track, run `music-info` on it, re-plan the scenes on
its sections (drops, breakdown) and show the updated storyboard timings before rendering. Apply what the user says (new lines, scenes, order, voice, mood, length, colours)
and show the updated storyboard again only if the change was big. Skip the question only when the
user already said "just make it", "no questions" or similar, or gave you a complete brief.

After the stills (Step 4) send the contact sheet image to the user too, before the long final
render, so they can stop it early.

### Step 3: write `brief.json` (section 4), then voice

```bash
python "$ENGINE/run.py" voice   <PROJECT>      # neural TTS, word timings, key words placed
```

It prints where every key word lands. If two lines overlap, shorten a line or move an anchor.

### Step 4: stills, look, fix

```bash
python "$ENGINE/run.py" stills <PROJECT>                    # ~20 key moments + build/stills/sheet.png
python "$ENGINE/run.py" stills <PROJECT> --times 3.2,9.8    # specific moments
python "$ENGINE/run.py" sheet  <PROJECT> --every 0.5        # whole film, one frame per 0.5 s
```

Read `sheet.png`. Check like a director: is the light always alive, is every hand-over continuous,
does text sit still when it must be read, nothing overlaps, nothing is cut off at the frame edge
(especially 9:16), colours are on brand, no placeholder text, no wrong product name. Fix the brief
(data, weights, colours) and re-render the stills. Two or three passes is normal.

### Step 5: preview, then final

```bash
python "$ENGINE/run.py" preview <PROJECT>   # 960x540 @30, full mix  -> out/<name>_preview.mp4
python "$ENGINE/run.py" final   <PROJECT>   # 1920x1080 @60, full mix -> out/<name>_final.mp4
python "$ENGINE/run.py" check   <PROJECT>   # frames, duration, LUFS, true peak, stutter spikes
```

Renders run in parallel 60-frame segments and resume after a crash: re-running skips finished
segments. After a change, delete `build/segs_final` (or pass `--only 9-12` to redo just the
segments around 9-12 s).

Renders are machine-safe by default: the worker count is sized to the RAM that is free right now
(about 0.9 GB per 1080p worker, 0.5 GB at preview size) while keeping 12 % of RAM for the user's
apps, capped at half the CPU cores; workers run at below-normal priority and each ffmpeg encoder
uses 2 threads. On a busy 16 GB laptop that can mean 1-2 workers (a 30 s 1080p60 film then takes
15-25 minutes); with more free RAM it scales up. `--workers N` sets a ceiling, never more than
fits. Tell the user up front how long the final will take, and always run it in the background.

### Step 6: hand over

Give the user the MP4 path(s), the beat sheet you used, and anything you would flag. Offer the
obvious variants in one line (9:16 cut, dark version, 15 s cut, another mood). Each is a brief edit.

---

## 3. Making every film unique

The engine is a vocabulary, not a template. Two films should never look or sound alike.

**Before directing, run `python "$ENGINE/run.py" history`.** It lists the creative choices of the
last films rendered on this machine (format, ending, opening, hook, stage, mood, voice, scenes).

**First pick a format: the shape of the whole film.** Never use the same format as the previous
film unless the user asks. A format only fills in what you leave out of the brief, so you can
still override any part of it.

| format | shape | fits |
|---|---|---|
| `journey` | logo slam -> one light does every job -> lights race back into the logo | productivity, storage, multi-feature SaaS |
| `keynote` | cold open on a line -> real UI with zooms -> understated logo reveal (no impact) | polished apps, AI, B2B, anything Apple-like |
| `launch` | logo hit up front -> whips and cuts on the beat -> hard cut to the logo | launches, new versions, dev tools, gaming |
| `story` | cold open on a human tension -> the product resolves it -> the name is written by light | health, education, non-profit, family, wellness |
| `showcase` | cold open on the product itself -> details and specs -> logo revealed by light | hardware, cars, physical products, luxury |
| `reel` | 9:16, fast, captions on, whips -> hard cut to the logo | Reels / TikTok / Shorts cut-downs |

Endings on their own: `"ending": "race" | "reveal" | "cut" | "write"`. Then change at least two
more axes from the most recent film unless the user asked for the same look:

| axis | options |
|---|---|
| opening | `slam` (logo first, then the product) or `cold` (straight into the product, logo revealed only at the end, like most Apple ads) |
| hook | `converge` (lights from the edges) or `trace` (one light draws the silhouette) |
| stage | light, dark, or a brand-coloured dark (`stage.color` = a deep brand shade) |
| mood | hype, uplifting, dark, cinematic, chill, house, acoustic, minimal (each also varies per product) |
| voice | different voice / gender / accent; or no voice with captions |
| tempo | `bpm` 84-128 (slower = premium and calm, faster = energy) |
| pacing | scene `weight`s, 3-6 acts, `duration` |
| transitions | light hand-overs, morphs, `whip`, `cut`, `wipe`, `push`, `shape` |
| scene mix | rebuilt UI, real screenshots, statements, features, product shots, custom scenes |

The score is seeded from the product name, so even two films in the same mood get different
chord progressions, keys, lead instruments, drum patterns and drum-kit tuning. The sound effects go
through a per-product palette too (below). Set `music.seed` / `sfx.seed` to re-roll.

**Signature: make the light belong to the product.** `"signature"` shapes every hand-over, the peel,
a cold-open hook and the end-card line, and gives the `signature` scene its hero moment, with
sounds on the motif's events:

| signature | the light becomes | fits |
|---|---|---|
| `ecg` | a heartbeat trace that beeps like a monitor | health, care, fitness, medical, wellbeing |
| `wave` | a sound wave | music, audio, voice, podcasts, calls |
| `circuit` | PCB traces with 45-degree corners that tick | dev tools, AI, hardware, infra, security |
| `route` | a winding road with pins | maps, travel, delivery, logistics, cars |
| `ripple` | a decaying ripple | messaging, social, notifications |
| `rise` | a climbing market line | finance, analytics, growth, sales |
| `spark` | a zig-zag spark | creative tools, design, writing, education |
| `none` | soft curves | anything where a motif would be a stretch |

If none fits, invent the product's own motif in `custom_scenes.py` (a coffee app's steam curl, a
fashion brand's stitch line) rather than forcing one.

### What fits what (read this before choosing effects)

Every effect means something. Use it only where its meaning is the product's meaning. Mix and
match freely within a row; borrow from a neighbouring row when the product sits between two.

| product kind | format | signature | stage / backdrop | scenes that fit | transitions | music / SFX | avoid |
|---|---|---|---|---|---|---|---|
| health, care, therapy, medical training | `story` | `ecg` | light or warm dark / `blobs` | `screen` (real UI), `card` quote, `signature`, `phone`, `devices` | light, cut | acoustic or uplifting / organic | mascots, `diorama`, confetti, glitchy `digital` SFX |
| video editors, design, creative tools | `launch` or `journey` | `spark` | dark studio / `grid` | `diorama` with mascot, `montage`, `type` build/giant, `tunnel` of templates | whip, cut, push | hype or house / punchy | slow `keynote` pacing, rings |
| AI assistants, copilots, chat | `keynote` | `circuit` or `none` | dark / `rings` | `chat`, `morph` (ask -> answer), `type` swap/cloud, `screen` | light, wipe, cut | minimal or uplifting / clean or digital | confetti, cartoon mascots |
| decision, research, legal, finance, B2B trust | `story` or `keynote` | `circuit` or `rise` | near-black + one brand colour / `rings` | `screen` real UI with highlights, `orbit` (the team), `morph` (a claim gets its source), `grid` (the bench), `card` quote, `type` swap/cloud | cut on drops, push, wipe, light | the user's track or cinematic / clean | confetti, mascots, `giant` gimmicks, bubbly `soft` SFX |
| integrations, automation, infra | `launch` | `circuit` | dark / `grid` | `orbit` converge, `grid`, `flow`, `chart` | whip, cut | dark or minimal / digital | slow cards |
| feedback, roadmap, community SaaS | `journey` | `ripple` | light / `ripple` | `type` cloud/giant/swap, `orbit`, `features`, `success` | light, whip | uplifting / clean | heavy cinematic braams |
| productivity, notes, storage | `journey` | `none` or `spark` | light / `blobs` | `upload`, `search`, `share`, `devices`, `morph` | light hand-overs, morphs | uplifting / clean | dark stages, `montage` |
| consumer apps, lifestyle, fitness | `reel` or `launch` | `rise` or `ripple` | brand-coloured / `horizon` | `phone`, `success`, `montage`, `type` build | whip, cut | hype or house / punchy or soft | long `screen` walkthroughs |
| hardware, cars, physical products | `showcase` | `route` or `none` | dark / `horizon` | `product` with specs, `montage` (graded), `type` teaser | cut, light | cinematic or dark / cinematic or mechanical | UI scenes, confetti |
| launches, "coming soon" teasers | `launch` | `spark` | dark / `horizon` | `type` teaser, `montage`, `giant` | cut on every bar | hype / punchy | voice-over-heavy scripts |
| education, kids, non-profit | `story` | `spark` | light / `blobs` | `card`, `phone`, `success`, `type` tilt | light | acoustic / organic or soft | dark trailer music |

How to read the meanings:
- `orbit` = many voices or inputs meeting one thing (a board of agents, integrations, a team).
  With `converge` they collapse into the product: "they all go after your pitch".
- `morph` = one thing becoming its answer (a question to a sourced card, a prompt to a page).
- `tunnel` = abundance (fifteen archetypes, a template library, a catalogue).
- `grid` = a bench, a toolbox, integrations: things you can call on.
- `success` + confetti = a personal win. Joyful products only; never health, money, law or grief.
- `diorama` + mascot = playful craft. Creative tools, editors, games, dev tools with a sense of
  humour; never a serious or clinical product.
- `type` swap = framing the problem; `cloud` = a vocabulary (rulings, features, tags);
  `teaser` = anticipation right before a drop; `build` + logo icon = the name, said once.
- Real UI (`screen`, `montage` of captures, `tunnel` of cropped cards) beats rebuilt UI whenever
  the user's product has a site or screenshots. Rebuilt scenes are for the concept around it.

- Different products get different **scenes, order, stage, hook, transitions, mood and pacing**.
  A dev tool: dark, trace hook, `statement` -> `chat` -> `flow` (whip) -> `chart`, mood dark.
  A notes app: light, converge hook, `upload` -> `search` -> `share` -> `devices`, uplifting.
  A car: dark, trace hook on the badge, `product` with specs -> `statement` (cut) -> `phone`
  (the companion app), mood epic.
- Use the product's **own words** from the site (headings, CTAs) for UI labels, search queries,
  node names and chat prompts. Rename the folder in `upload`, the query in `search`, the
  people in `share`.
- **Weights** (`"weight"` per act) change pacing: give the hero job more time.
- **Write a custom scene** when the product's core job is not in the table (a map route, a
  checkout, a code editor, a calendar). See section 6. It takes ~80 lines with the kit.

---

## 4. brief.json

```jsonc
{
  "aspect": "16:9",              // "16:9" | "9:16" | "1:1" | "4:5"
  "duration": 21.8,              // seconds; the hit lands on a downbeat, end card is 2 s after the lockup
  "bpm": 100,                    // the picture grid; set it to the track's BPM when using a track
  "format": "journey",           // journey | keynote | launch | story | showcase | reel (fills the rest)
  "energy": "normal",            // hype | normal | calm: beat punches, drop impacts (default from format)
  "signature": "none",           // ecg | wave | circuit | route | ripple | rise | spark | none
  "sfx": {"palette": "clean"},   // clean | punchy | organic | soft | digital | cinematic | mechanical
  "ending": "race",              // race | reveal | cut | write
  "opening": "slam",             // "slam" (logo first) | "cold" (logo only at the end)
  "hook": "converge",            // "converge" | "trace"  (with "cold", lights converge into scene 1)
  "stage": {"mode": "light", "color": "#FAFAFB", "strength": 0.16,    // dark: {"mode": "dark", "color": "#06070B"}
            "backdrop": "blobs"},        // blobs | horizon (a planet rim) | rings (a round table) | ripple | grid (a floor)
  "brand": {
    "name": "Prism",             // used for defaults ("Search in Prism") and the output file name
    "wordmark": "Prism Drive",   // text written in the lockup
    "wordmark_weight": 400,
    "wordmark_color": "#1F1F1F", // optional; the brand's wordmark colour
    "wordmark_font": "body",     // "display" writes the wordmark in font.display (a serif brand, say)
    "logo": "assets/logo.svg",   // SVG best; PNG ok
    "facets": "auto",            // "auto" | "paths" (one per SVG shape) | "colors" | "none"
    "facet_count": 2,            // optional, for "colors": how many colour facets
    "primary": "#3186FF",        // optional; UI accents
    "palette": ["#3186FF", "#0EBC5F", "#FEC700"]   // fallback lights for monochrome logos
  },
  "lights": ["#0EBC5F", "#FEC700", "#3186FF"],     // optional: one per act, in order
  "font": {"family": "Inter", "display": "Oswald"},  // body + condensed display face (cards); any Google Fonts
  "ink": {"text": "#1F1F1F", "secondary": "#5F6368"},
  "acts": [
    {"scene": "upload", "data": {...}, "weight": 2.3, "light": 0, "transition": "light"}
  ],
  "vo": [
    {"text": "Every file.", "key": "file", "at": "act1.w1"},
    {"text": "One place!", "key": "One", "at": "act1.drop", "role": "drop"},
    {"text": "Found in seconds.", "key": "seconds", "at": "act2.key"},
    {"text": "Prism Drive.", "key": "Prism", "at": "hit+0.3", "role": "brand"}
  ],
  "voice": {"engine": "edge", "voice": "en-US-AvaMultilingualNeural", "rate": "+4%", "pitch": "+0Hz"},
  "music": {"mood": "hype"},     // see section 5
  "captions": "auto",            // "auto" (on for 9:16) | "on" | "off"
  "end_card": {"style": "credit", "top": "made by", "name": "Your Name",
               "small": "Spec ad · not affiliated with Acme"},
  // or {"style": "cta", "headline": "Start free today", "url": "acme.com"}
  // or {"style": "tagline", "headline": "A verdict, not a compliment.", "url": "acme.com", "hold": 5}
  //    (the line is the hero in the display face, the address small under the light; "hold" = seconds)
  "output_name": "prism_spec_ad"
}
```

**Anchors** for `vo[].at` (and anything timed): `slam`, `lockup`, `drop`, `race`, `hit`, `end`,
`actN` (scene start), `actN.key` (each scene's key moment), plus scene-specific ones:
upload `w1 w2 w3 drop`, search `type`, share `push`, flow `start`, chat `send`, screen `h1 h2..`,
product `sheen s1 s2..`, phone `tap`, statement `written`. Offsets work: `"hit+0.3"`, `"act2.key-0.1"`.

### Scene data (all optional; sensible defaults exist)

- `upload`: `folder`, `uploading` ("Uploading 1,284 items"), `count_label`, `done` ("Backed up"),
  `waves` ([{"kinds": ["doc","sheet"], "count": 8} x3]), `drop_count`.
  File kinds: doc sheet slides pdf photo video code audio design link task chat.
- `search`: `placeholder`, `query`, `results` ([{kind, title, meta}] x3), `select` (index).
- `share`: `title`, `file`, `kind`, `page` ("doc" | "canvas" | "board"), `button` ("Share"),
  `people` ([{name, color}] x3, fictional names).
- `devices`: `chip` ("Synced"), `screen` / `tablet` / `mobile` (screenshot paths for the screens).
- `flow`: `nodes` ([{label, kind}] 3-5), `result` ("Done").
- `chart`: `series` (numbers), `value` ("+38%", "$1.2M"), `label`, `title`.
- `chat`: `placeholder`, `prompt`, `answer` (3-4 short lines), `scale` (1.0-1.4; bigger for a
  cold open where the chat is the whole frame).
- `screen`: `image` (a real screenshot or video frame), `crop` ([x0, y0, x1, y1] to cut the UI out
  of a larger frame), `url` (shown in the browser bar), `frame` ("browser" | "bare"), `enter`
  ("tilt" | "rise" | "slide"), `tilt` (entry yaw in degrees; sign picks the side), `highlights`
  ([{rect: [x, y, w, h] in the image's pixels, label, zoom: 1.2-1.6, click: true}]). The camera
  pushes in on each highlight, the rest of the UI dims, the light traces the detail and the label
  pops in. Two to three highlights per screen; vary `enter` and `tilt` between consecutive screens.
- `statement`: `lines` (1-3 short lines), `highlight` (one word), `size`, `weight`.
- `features`: `items` ([{title, desc, kind}] 3-6).
- `phone`: `image` (app screenshot, optional), `title`, `rows`, `tap`, `notify` ({title, body}).
- `product`: `image` (transparent PNG best), `specs` ([{value, at: [x, y] 0-1 on the image}]).
- `diorama`: `image` + `crop` (the UI texture; omit for a generated dark editor with timeline
  lanes), `camera` ("swoop" | "flyover" | "orbit" | "reveal" | "follow"), `mascot`
  ({color, size} or false), `path` ([[u, v] ...] 0-1 on the image: the hops; keep them on a band of
  the UI near the camera), `lanes` ([{v, u0, u1, color}] glowing bars that grow behind the mascot),
  `playhead` (true), `title` ({text, rect: [u0, v0, u1, v1], color} glowing on the UI, like a
  title on the app's preview), `table` (hex), `key_light` (hex), `dof` (0-1.5). Anchors: `go`,
  `l1 l2 ..` (each landing), `key` (the last landing).
- `card`: `layout` ("headline" | "quote" | "stat"), `lines` (2-3 short lines; UPPERCASE for
  headlines), `highlight` (words drawn in the light colour), `value` (stat: "390", "4", "$1.2M"),
  `by` (quote attribution). The display face is `font.display` (any Google Fonts family; default
  Oswald).
- `signature`: `lines` (1-2 short lines above the trace), `highlight` (one word in the light colour),
  `kind` (a motif other than the film's `signature`). Anchors: `key`, `e1 e2 ..` (each heartbeat,
  node or pin).

---

## 5. Sound (the user chooses; these are the defaults you pick for them)

- `music.mood` (each one is re-composed per product from a seed: progression, key, lead, drums):
  - `hype`: trailer-pop. 808 drop, trap hats, supersaw stabs, sidechain pump.
  - `uplifting`: clean and bright, Apple-keynote energy. Four-on-the-floor, pumping chords.
  - `dark`: car-ad. Half-time, distorted 808, toms, braams, big snare.
  - `cinematic` (= `epic`): taiko ostinato, string runs, brass braams, no 808.
  - `chill` (= `lofi`): swung drums, FM electric piano, vinyl crackle, warm sub.
  - `house`: four-on-the-floor, offbeat bass, organ stabs, shakers.
  - `acoustic`: plucked guitar (Karplus-Strong), snaps, shaker, warm pad. Human and warm.
  - `minimal`: glassy bells, clicks, sub pulses, lots of air.
  - `none`: SFX and voice only.
  Pick by product and the user's words: "Apple style", "clean", "inspiring" -> uplifting;
  "car ad", "Nike", "intense" -> dark; "trailer", "epic", "launch" -> cinematic; "hype", "TikTok",
  "bass drop" -> hype; "calm", "cozy", "lofi" -> chill; "fashion", "club", "fintech" -> house;
  health, education, food, family, non-profit -> acoustic; dev tools, AI, B2B, security -> minimal.
  `music.seed` (any number) re-rolls the song; `music.transpose` fixes the key.
- `sfx.palette` (defaults from the mood; every product also gets its own seeded variation of pitch,
  colour and room):
  - `clean`: bright, tight, polished UI sounds (consumer apps, SaaS).
  - `punchy`: saturated hits, bright whooshes (launches, hype).
  - `organic`: wooden clicks, marimba pops, warm room (health, food, education, wellness).
  - `soft`: bubbly pops, rounded and quiet (calm, lifestyle, kids, notes).
  - `digital`: bit-crushed zaps and ticks (dev tools, AI, crypto, gaming).
  - `cinematic`: low booms, big rooms (cars, hardware, epic launches).
  - `mechanical`: servo whines under the moves (hardware, robotics, automotive).
- `music.file`: the user's own track (mp3/wav/m4a). The whole film then runs on the song:
  - `python "$ENGINE/run.py" music-info track.mp3` prints the BPM, the downbeats and a bar-by-bar
    map of the song (loudness, low end, and where it builds, drops and breaks down). Read it.
  - Beats come from a tracker that follows the song's real tempo drift, not a metronome, so cuts
    stay on the beat for the whole minute. Every scene boundary snaps to a beat of the song, and
    kinetic type lands on the narration's words.
  - `music.drop_at`: the drop (track seconds) that lands on the logo hit; it snaps to the
    nearest detected downbeat, so use the map's DROP rows. `music.hit_at`: the film time of that
    hit (default: 4.4 s before the end). The track is offset by `hit_at - drop_at`.
  - Then place scenes on the song's sections in film time (track time + offset): the product
    reveal on the first DROP, the reflective or "how it works" beat in the breakdown, the logo on
    the last DROP. Set act `weight`s to the seconds you want; boundaries then fall where planned.
    Print the timeline to check: `python -c "from adkit.film import Film; f = Film('<PROJECT>',
    0.25, 30); [print(s.spec['scene'], round(s.a, 2), round(s.b, 2)) for s in f.scenes]"`
    (run it from the engine folder).
  - Duration is the hit + 2.4 s + `end_card.hold`. The track is thinned before the hit and fades
    under the end card.
- `music.transpose` (semitones), `music.gain_db`, `sfx_gain_db`.
- Voice: `voice.engine` = `edge` (free Microsoft neural voices, needs internet), `elevenlabs`
  (needs `ELEVENLABS_API_KEY`; `voice.voice_id`, `voice.model` "eleven_v3"; audio tags like
  `[upbeat]` in `voice.prefix` or the text), or `none`. Good edge voices:
  `en-US-AvaMultilingualNeural` (warm, premium), `en-US-AndrewMultilingualNeural` (male, calm),
  `en-US-EmmaMultilingualNeural`, `en-US-BrianMultilingualNeural`, `en-GB-SoniaNeural`,
  `en-IN-NeerjaNeural`, `en-AU-NatashaNeural`. `voice.rate` "+4%" reads livelier.
- The mixer ducks music -8 dB under every word and -6 dB under hits, keeps the voice >= 15 dB over
  the music at 1-4 kHz, and masters to -14 LUFS, true peak <= -1 dBTP (checked after AAC encoding).
- Stems are written to `build/stems/` (music, sfx, voice) if the user wants to remix.

---

## 6. Custom scenes (when the library does not fit)

Create `<PROJECT>/custom_scenes.py`; it is loaded automatically and can register new scenes:

```python
from adkit.scenes import Scene, register, rrect_points
from adkit.core import Path, ramp, ease_out, ease_in_out, smooth, spring_in, lin
from adkit import ui

@register("route")
class Route(Scene):
    weight = 1.4
    def setup(self):                      # times: self.a, self.b, self.land (light arrives)
        f = self.f                        # the Film: f.CX, f.CY, f.DW, f.DH, f.theme, f.palette
        self.path = Path.beziers([((f.CX - 600, f.CY + 200), (f.CX - 200, f.CY - 300),
                                   (f.CX + 200, f.CY + 300), (f.CX + 600, f.CY - 150))])
        self.key = self.a + 0.75 * self.L
    def entry(self): return tuple(self.path.at(0))
    def exit(self):  return tuple(self.path.at(1))
    def anchors(self): return {"key": self.key}
    def cues(self): return [dict(t=self.key, kind="success", gain=-15)]
    def draw(self, cv, t):
        out = 1 - smooth(ramp(t, self.b - 0.1, self.b + 0.25))
        u = ease_in_out(ramp(t, self.land, self.key), 1.8)
        cv.light_poly(self.path.span(0, u, 120), lin(self.color), 0.5 * out, width=3)   # lit trail
        cv.light_path(self.path, u, tail=0.4, color=self.color, width=4.5, energy=1.1 * out)
        card = self.sp("card", lambda: ui.card(300, 90, 20, self.ppd, self.fonts, self.T))
        if t > self.key:
            p = self.path.at(1)
            cv.draw(card, p[0], p[1] - 90, scale=spring_in(t - self.key), alpha=out)
```

Kit reference:
- `cv.draw(sprite, x, y, scale, rot, alpha, anchor, blur, wipe, dim, clip, band, grad, flip)`,
  `cv.draw_quad(sprite, 4 points)` (3D), `cv.draw_motion(sprite, fn, t, shutter)` (motion blur).
- `cv.light_path(path, head, tail, color, color_tail, width, energy, occlude)` (a travelling light),
  `cv.light_poly(points, colors, intensities, width)` (a static lit line), `cv.light_dot`.
- `Path.beziers([...])`, `Path.through(a, b, bend)`, `Path(points, closed)`, `.at(u)`, `.span(u0, u1)`.
- `ui.card/pill/chip/folder/file_card/result_row/doc_page/cursor/name_flag/avatar/laptop/handheld/
  node_card/chart_card/browser_frame/chat_bubble`, and `ui.Painter` for anything else
  (`rrect`, `circle`, `line`, `poly`, `text`, `.sprite()`).
- Text: `self.text(s, size, weight, color)` returns a sprite; `sprite.baseline` aligns text;
  `sprite.meta["advance"]` gives per-character x for typing reveals.
- Easing: `smooth`, `ease_out` (arrivals), `ease_in` (exits, short moves only), `ease_in_out`,
  `spring_in` (pops), `damped`, `pulse`. Never put a long move on a steep `ease_in`.
- Sound cue kinds: whoosh_hit whoosh_short whoosh_soft whoosh_fast swoosh swirl digital impact
  transient ring tick key click select pop pop_soft pop_low success confirm sparkle shimmer send
  air riser_end riser_short breath. Each cue: `dict(t=, kind=, gain=dB, pan=-1..1, pitch=)`.

---

## 7. Craft rules (check these on every contact sheet)

1. Light comes from behind or IS the element (a bar, a caret, an outline); never a flat neon stroke
   slapped on top. One light per job; hand-overs recolour along the tail.
2. Text holds still while it must be read. Max ~5 words of headline on screen.
3. Snaps snap (2-3 frames); long moves use smooth/ease_out. "It lags" = a long ease_in somewhere.
4. Arrange, don't scatter: rings, rows, grids and arcs read as designed.
5. Things persist: containers morph (row -> page -> laptop screen) instead of popping.
6. Everything that flies has motion blur; everything that lands has a sound.
7. On white: no white light cores, no white flashes, device screens are real holes.
8. In 9:16 keep everything inside ~90 % of the width; captions sit in the lower third.
9. Fictional people, files and numbers only. No real faces. No other companies' logos.
10. Spec ads for brands the user does not own: keep the official logo untouched and put
    "Spec ad · not affiliated with <Brand>" on the end card. For the user's own product use the
    `cta` end card.

## 8. Troubleshooting

- Logo facets look wrong: try `"facets": "colors"` (or `"paths"`), or `"none"` for one layer;
  `brand.facet_lights` overrides the light colours per facet.
- SVG fails to rasterise: install Chrome/Edge/Chromium or `pip install cairosvg`, or use a PNG.
- No voice: edge-tts needs internet; otherwise set `"voice": {"engine": "none"}`.
- A scene feels cramped: raise its `weight` or the film `duration`.
- Re-render one part only: `final <PROJECT> --only 9-12` after deleting those segments, or delete
  `build/segs_final` for a full re-render.
