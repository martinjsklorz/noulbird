# noulbird

**You vs a language model at Flappy Bird.** A real-time demo of
[laya-mlx](https://pypi.org/project/laya-mlx/) — native MLX inference for Laya
typed decision models on Apple Silicon.

The name comes from the *noul*: the probability a Laya model returns when you
ask it whether something is true. Every flap the AI bird makes is one, served
by the model's `systemone` endpoint — fast, reflexive judgements, made under a
deadline rather than deliberated.

You fly the blue bird; a locally-running classifier flies the amber one. Both
face the same pipes at the same time, and when one crashes the other keeps
going.

Everything runs on your machine — no API key, no account, no waitlist, no
network calls. `pip install`, run, play.

<!-- Add a screenshot or GIF here. -->

## What this demonstrates

Most LLM demos are turn-based: type a prompt, wait, read a reply. This one puts
a model inside a 60fps game loop, where it has to make a decision every ~120ms
while a bird falls out of the sky. That constraint surfaces things a chat demo
never will.

Laya models are *typed decision models*. Rather than generating text, they
answer structured questions about a state and return numbers:

| Type | Returns | Example |
| --- | --- | --- |
| `noul` | Probability a statement is true, 0–1 | "Is the bird below the gap?" → `0.80` |
| `choice` | One option plus probabilities for all | "Billing, technical, or sales?" → `billing` (96%) |
| `score` | A graded value on a scale | "How angry is this?" → `1.44 / 3` |

Because the output is a number rather than a token stream, there is nothing to
parse, no JSON to coax out of a model, and no retry loop when it returns prose
instead of a verdict. That is what makes it viable at 120ms per decision.

## What it takes to make this work

Three properties of the model shape the design. They are worth knowing before
you build anything real on it, and each is visible in the code.

**Ask what the model can perceive, not what to do.** Asked "should the bird
flap?", it returns roughly the same value regardless of the bird's position.
Asked "is the bird below the gap?", it answers sharply and correctly. The game
asks only perceptual questions and derives the action itself, in
`buildRequest()`.

**Describe the world in prose, not numbers.** Given a velocity of `-8` or
`+8`, the answer to "is the bird falling?" barely moves. Given "moving upward
fast" or "moving downward fast", it separates cleanly. So `buildState()` emits
sentences:

> The bird is below the safe gap. The bird is moving downward fast.

**Treat inference as slower than your loop.** A decision takes tens to hundreds
of milliseconds and varies with machine load. The AI issues one request at a
time and waits for each reply. A fixed-rate timer would queue requests faster
than they complete, and the bird would act on an increasingly stale world.

## Requirements

- Apple Silicon Mac — `laya-mlx` is built on [MLX](https://github.com/ml-explore/mlx)
- Python 3.9+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open <http://localhost:8786>. The first run downloads the model
(`convaiinnovations/laya`); the status dot beside the title turns green when it
is ready.

```
python server.py --port 8080          # different port
python server.py --host 0.0.0.0       # expose on the local network
python server.py --model some/other   # a different model
```

## Playing

| Action | Control |
| --- | --- |
| Flap | `Space`, `↑`, click, or tap |
| Start a round | Any flap, or **Start round** |
| Reset the match tally | Click the `0–0` counter |

Enable **Decision stream** to watch the model think in real time:

```
#12 0.685 FLAP y:340 vy:+1.5 off:70 [m0.82 h0.55] 150ms
```

That is the decision number, the resulting value, the action taken, the bird's
state, the two halves of the decision, and the round-trip latency. `off` is the
offset from the centre of the next gap (positive means below it); `m` and `h`
are the model's reading and the deterministic predictor's, so you can see which
one is responsible for any given flap.

## How the AI flies

Every ~120ms the game describes the world and asks two questions about it:

```
"The bird is below the safe gap. The bird is moving downward fast."

  below_gap: 0.80  ─┐
                    ├─ 0.75·below + 0.25·falling ─┐
  falling:   0.86  ─┘                             ├─ average → flap if > 0.5
                                                  │
  heuristicFlap(): 0.55 ──────────────────────────┘
```

The model's reading is averaged with `heuristicFlap()`, a deterministic
controller that projects the bird forward and asks whether gliding would fall
short. This is not redundancy for its own sake: the model resolves position
into about five coarse buckets, which cannot distinguish a 15px error from a
1px one near the gap. The blend keeps the model's judgement and recovers the
precision.

`heuristicFlap()` also flies alone when the model is unreachable, so the game
stays playable with the server down.

## Using laya-mlx directly

The server wraps the library in a small HTTP endpoint, so you can try questions
without touching the game. Several are evaluated against one state in a single
call:

```bash
curl -X POST http://localhost:8786/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "A support email: \"My invoice is wrong and I have emailed three times with no reply.\"",
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Is this urgent?",
        "criteria": {"true": "urgent", "false": "not urgent"}
      },
      "topic": {
        "type": "choice",
        "instructions": "What is this about?",
        "criteria": ["billing", "technical", "sales"]
      },
      "anger": {
        "type": "score",
        "instructions": "How angry is the sender?",
        "criteria": ["calm", "annoyed", "angry", "furious"]
      }
    }
  }'
```

```json
{
  "answers": {
    "urgent": { "noul": 0.5444, "confidence": 0.5444 },
    "topic": {
      "choice": "billing",
      "confidence": 0.8331,
      "probabilities": { "billing": 0.9624, "technical": 0.0261, "sales": 0.0115 }
    },
    "anger": {
      "score": 1.4421,
      "confidence": 0.2723,
      "probabilities": { "0": 0.033, "1": 0.5936, "2": 0.2716, "3": 0.1018 }
    }
  },
  "model": "laya-mlx",
  "latency_ms": 3451.6
}
```

Two things worth knowing: `score` criteria are an ordered list of labels, not a
numeric range; and on some checkpoints the library warns at load time that
`confidence` is uncalibrated. The game relies only on `noul`, which is
unaffected.

Or skip the server entirely:

```python
import laya_mlx as laya

agent = laya.load("convaiinnovations/laya", dtype="float16")
result = agent.predict(
    "The bird is below the safe gap. The bird is moving downward.",
    {
        "below_gap": {
            "type": "noul",
            "instructions": "Is the bird below the safe gap?",
            "criteria": {
                "true": "The bird is below the safe gap",
                "false": "The bird is above the safe gap",
            },
        }
    },
)
print(result["answers"]["below_gap"])
```

`GET /v1/health` reports whether the model has finished loading.

## Tuning

The constants at the top of the `<script>` block in `index.html` control
difficulty. One relationship is worth knowing first.

`PIPE_SPEED` and `PIPE_INTERVAL` determine how many decisions the AI gets per
pipe, because decisions arrive on a wall clock rather than per frame. Slowing
the pipes down while leaving `AI_INTERVAL` alone *reduces* that budget and
makes the AI play worse — the opposite of what "make it easier" implies. To
genuinely ease things off, lower `GRAVITY` and `FLAP_VEL` together, or widen
`PIPE_GAP`.

## Project layout

```
index.html        Game, UI, and AI control loop. No build step, no dependencies.
server.py         FastAPI server: loads the model, serves the page and the API.
requirements.txt  Python dependencies.
```

The page is a single self-contained file using system fonts, so it works
offline and can be served from anywhere — though the AI bird needs `server.py`
on the same origin to reach the model.

## Licence

This demo is MIT; see [LICENSE](LICENSE). `laya-mlx` is Apache-2.0 and is not
affiliated with this project.
