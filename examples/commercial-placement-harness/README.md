# Commercial placement harness

Public companion for the Recohut docs page
[A commercial placement harness](https://docs.recohut.com/use-cases/commercial-placement-harness).

The page's claim is that a send stays unwritten until a person approves that
exact action. This directory is the desk rule behind that claim: a span must
occur in its source, appetite is code, and the outbox is keyed by the action id.

The labs run also used the private Recohut kit for the durable run, the wait,
and the audit journal. Those pieces are not in this repository. What is here is
enough to rerun the four host checks without a model and without a key.

## Run the checks

From this directory:

```bash
python test_desk.py
```

No third-party packages. The fixtures are synthetic.

## What a passing run shows

- A span that is not in the statement is rejected and the field is not stored.
- A tyre-recycling occupancy is a decline and cannot open a send.
- `propose_send` leaves the outbox empty. `approve` writes one row. Approving the same action id does not write another.
- The pending action id is still the one that was proposed.

The live `grok-4` counts in the article are not reproduced here. This directory does not call a model.
