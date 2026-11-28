# A read-only insurance service desk

Companion code for the Recohut docs page
[A read-only insurance service desk](https://docs.recohut.com/use-cases/insurance-service-desk).

One synthetic customer. A router may choose a lane. The host owns the customer
id, refuses a payment before any model call, and keeps a citation only when a
tool on that turn returned it. The records are seed data. Nothing is paid,
filed, or changed.

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

The live command reads `GROK_API_KEY` from the environment. It is not stored in
this repository. `GROK_MODEL` overrides the model name. The default is `grok-4.7`.

## Run

```bash
PYTHONPATH=src python -m insurance_desk ask "What is the status of claim CLM-1001?"
```

The process binds `cust-maria`. SQLite is created under `.local/` on first use.
That directory is not part of the repository.

## Tests

The five checks do not call a model:

```bash
PYTHONPATH=src python -m pytest tests/test_service_desk.py
```

They cover a payment that never reaches a model, a claims specialist that cannot
read policy or billing tools, escalate and clarify skipping the specialist,
tenant-filtered search, and a citation the tools did not return.
