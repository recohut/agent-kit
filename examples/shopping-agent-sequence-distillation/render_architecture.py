"""Generate an automatic topology view separate from the edited Draw.io figure."""

import argparse
from pathlib import Path

from diagrams import Cluster, Diagram, Edge, Node


def node(label, *, shape="box"):
    return Node(label, shape=shape, fixedsize="false", labelloc="c")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with Diagram(
        "Sequence distillation topology",
        filename=str(output_dir / "distillation-topology"),
        outformat=["dot", "svg"],
        direction="LR",
        show=False,
    ):
        traces = node("Teacher rollouts")
        task = node("Held-out product task")
        result = node("Scored evaluation")
        with Cluster("Distillation and benchmark harness"):
            scorer = node("Native scorer")
            encode = node("Assistant-only SFT")
            qlora = node("4-bit QLoRA", shape="cylinder")
            react = node("ReAct tool loop")
            loopback = node("Loopback inference")
        traces >> Edge(label="12 traces") >> scorer >> encode >> qlora
        task >> react >> loopback >> result
        qlora >> Edge(label="LoRA adapter") >> loopback


if __name__ == "__main__":
    main()
