<p align="center">
  <img align="center" src="docs/docs/static/img/pirborg_logo.png" width="460px" />
</p>
<p align="left">

# pirborg — Prompt Language Intermediate Representation

**pirborg** is an **Internal Representation for Prompts**, (PIR) for short. It gives you structured building blocks to **write prompt optimizers** and **dialect‑specific codegen** with the same discipline compilers enjoy: clear semantics, static checks, and clean lowering paths.

---

## Ambition

* Make writing **prompt optimizers** simple and safe.
* Provide a **stable IR** that tools can target and generate from.
* Offer **fine‑grained control** over what is optimizable (sections, inputs, slots, routes).
* Support **multi-call prompt graphs** (e.g., ReAct chains), **bounded loops**, and **selection/judging** patterns.
* Enable **extensible dialects** for code generation (back to DSPy, or to other runtimes such as Go).
* Ship a **textual format (PIR‑TXT)** and a **deterministic Python EDSL** with round‑trips between them.
* Supply **static analysis** and **lints** for template correctness and schema hygiene.

---

## Highlights

* **IR, not runtime:** A compact Prompt IR (PIR) with a textual form (**PIR‑TXT v1**) and a Python EDSL.
* **Deterministic authoring:** EDSL primitives ensure reproducible structures; templates default to strict semantics.
* **Loops & graphs:** Express bounded loops and prompt graphs with selection criteria and optional judges.
* **Optimizers:** Pluggable backends. **GEPA** works today; **DSPy** backend included; **OpenEvolve** and **ShinkaEvolve** on the roadmap.
* **Static checks:** Lints for unused sections, channel violations, incomplete switches, schema collisions, and more.
* **Template discipline:** Detect unused/unknown inputs, enforce placeholder usage, and produce actionable feedback.
* **CLI tooling:** `pirborg fmt | lint | diff | render` for **PIR‑TXT** files.

---


## Status

Early-stage software, APIs subject to change. Currently in the process of being open-sourced. Feedback is encouraged.

## Install (from source)

```bash
uv pip install pirbog
```



> GEPA backend requires the `gepa` package; DSPy backend requires `dspy-ai`. Install them as needed.

---

## Textual IR vs EDSL (side‑by‑side)

Below, the **same prompt** in PIR‑TXT and the Python EDSL.

### PIR‑TXT (v1)

```pirborg
pirborg.module @examples.with_sections v1 {
  pirborg.input %language : str {channel = "system", required}
  pirborg.input %question : str {channel = "user", required}

  pirborg.section @persona {
    channel = "system"
    optimizable
    description = "Sets assistant persona and tone."
    text = "You are concise and always respond in {{ inputs.language }}."
  }

  pirborg.section @format {
    channel = "system"
    description = "Response contract."
    text = "Return JSON with an `answer` field only."
    output = {key="answer", json_type="string", required=true, description="Primary answer text"}
  }

  pirborg.message "system" {
    emit.section @persona
    emit.literal "\n"
    emit.section @format
  }

  pirborg.message "user" {
    emit.input %question
  }

  pirborg.render { engine = "jinja2" strict_undefined }
}
```

### Python EDSL

```python
from pirborg.edsl import program, inp, section, output, predict

with program("examples.with_sections"):
    inp.str("language", channel="system")
    inp.str("question", channel="user")

    section(
        "persona",
        text="You are concise and always respond in {{ inputs.language }}.",
        channel="system",
        optimizable=True,
        description="Sets assistant persona and tone.",
    )
    section(
        "format",
        text="Return JSON with an `answer` field only.",
        channel="system",
        output=output("answer", description="Primary answer text"),
        description="Response contract.",
    )

    prompt = predict(
        system="{{ emit_section('persona') }}\n{{ emit_section('format') }}",
        user="{{ inputs.question }}",
        id="examples.with_sections",
    )
```

---

## Loops and selection

Bounded loops are compiled into a deterministic **PromptGraph**. Here are two common patterns:

### Best‑of‑N with a judge

```python
from pirborg.edsl import program, inp, predict, loop_unroll

with program("qa.boN"):
    inp.str("question", channel="user")

    base = predict(system="Be accurate.", user="{{ inputs.question }}", id="qa.base")
    judge = predict(
        system="Score candidate 0..1 for factuality and relevance.",
        user="Q: {{ inputs.question }}\nCandidate: {{ inputs.candidate }}",
        id="qa.judge",
    )

    bo5 = loop_unroll(
        name="bo5",
        max_iters=5,
        body=base,
        feed={"question": "inputs.question"},
        select={
            "kind": "argmax",
            "score_field": "score",
            "judge": judge,
            "judge_feed": {"candidate": "step.output.answer"},
        },
    )
```

### Bounded refine loop with early stop

```python
from pirborg.edsl import program, inp, predict, loop_unroll

with program("qa.refine"):
    inp.str("question", channel="user")
    base = predict(system="Be accurate.", user="{{ inputs.question }}", id="qa.base")

    refine = loop_unroll(
        name="refine",
        max_iters=3,
        body=base,
        feed={"question": "inputs.question + state.feedback_suffix"},
        state={"feedback_suffix": "''"},
        update={"feedback_suffix": "'\\nPlease correct: ' + step.output.critique"},
        select={"kind": "first_success", "predicate": "step.output.confidence >= 0.9"},
    )
```

---

## Optimizer backends

Use a shared API to optimize **just the parts you’ve marked** optimizable.

```python
from pirborg.optimizer import (
    optimize_prompt, EvalSuite,
)
from pirborg.optimizer.runners import CallableRunner
from pirborg.optimizer.scorers import ContainsSubstringScorer

# Your runner: how to execute messages against an LLM
runner = CallableRunner(lambda messages, instance, candidate: "some model output")

# Your scorer: how to score outputs
scorer = ContainsSubstringScorer()
evals = EvalSuite(runner=runner, scorer=scorer)

# Optimize with GEPA (requires 'gepa' installed)
result = optimize_prompt(
    prompt,            # a PromptSpec
    optimizer="gepa",  # or "dspy"
    trainset=[{"inputs": {"question": "2+2?"}, "label": "4"}],
    valset=None,
    evals=evals,
    max_metric_calls=50,  # via GEPAOptions
)

optimized = result.optimized_prompt
```

**Status**

* ✅ **GEPA** backend working (`optimizer="gepa"`).
* 🧭 **OpenEvolve** and **ShinkaEvolve** are on the roadmap.

You can also discover and register optimizers:

```python
from pirborg.optimizer import get_registered_optimizers, register_optimizer

print(get_registered_optimizers())  # ["dspy", "gepa", ...]

# register_optimizer("myopt", lambda **kw: MyBackend(...))
```

---

## Fine‑grained optimization control

Mark **sections**, **inputs**, **slots**, and even **routes** as optimizable:

```python
from pirborg import Section, Input, InputType, Slot, SlotOption

# Section toggles
Section(name="closing", text="Summarize...", optimizable=True, description="Closing checklist")

# Input toggles and hints
Input(name="segment", type=InputType.ENUM, enum=["new_user","power_user"], optimizable=True,
      optimizer_hints={"default": "new_user"})

# Slot with options and initial hint
Slot(
  name="resolution_mode",
  options=[SlotOption(id="guided", text="Step-by-step"), SlotOption(id="escalate", text="Escalate to human")],
  optimizable=True, optimizer_hints={"initial": "guided"}
)
```

**Routes** created by `switch` over enum inputs are exposed as optimizable atoms too, allowing optimizers to explore control‑flow choices.

---

## Template discipline & static analysis

* **Strict undefined** by default for templates.
* Detect **unknown** or **unused** inputs.
* Lints for structural issues.

```python
from pirborg.renderer import render_template
from pirborg.lints import lint_prompt

# Enforce unknown/unused inputs at render time
rendered = render_template(prompt, prompt.template.system_tmpl,
                           inputs={"language": "Swedish", "extraneous": 1},
                           enforce_unknown_inputs=True)  # raises on "extraneous"

# Run lints (unused sections, channel violations, incomplete switches, schema duplicates)
issues = lint_prompt(prompt)
for issue in issues:
    print(issue.code, issue.message)
```

This enables optimizers to surface **rich textual feedback** when templates violate contracts (e.g., missing placeholders), and lets compilers like GEPA adjust formats safely.

---

## Prompt graphs and tools

Connect multiple prompts and external tools with typed contracts:

```python
from pirborg import PromptSpec, GraphNode, PromptGraph, EdgeBinding, ExternalNode

# Nodes
qa = GraphNode(id="qa", prompt=prompt, entry=True)
grader = GraphNode(id="grader", prompt=judge_prompt)

# External tool with strict I/O shape
web = ExternalNode(id="browser", inputs_schema={"query":"string"}, outputs_schema={"snippets":"array"})

# Edges: route JSON outputs to downstream inputs
edges = [
    EdgeBinding(from_node="qa", json_path="$.answer", to_input="grader.inputs.candidate"),
]

graph = PromptGraph(id="react.example", nodes=[qa, grader], edges=edges, external_nodes=[web])
```

---

## CLI (pirborg‑TXT tools)

```
pirborg fmt   path.pirborg                    # format in place
pirborg lint  path.pirborg                    # validate structure
pirborg diff  left.pirborg right.pirborg         # unified diff
pirborg render path.pirborg --inputs in.json  # render with JSON inputs (and --slots slots.json)
```

---

## Round‑tripping

Convert between PromptSpec and PIR‑TXT:

```python
from pirborg.text import module_from_prompt_spec, format_module, parse_module, prompt_spec_from_module

module = module_from_prompt_spec(prompt)
text = format_module(module)           # -> string in PIR‑TXT
parsed = parse_module(text)            # -> TextModule
roundtrip = prompt_spec_from_module(parsed)  # -> PromptSpec
```

---

## Roadmap

* Pipelines like **DSPy → PIR → Optimizer → DSPy** (refine authoring with learned sections).
* **DSPy → PIR → Optimizer → Go code** (codegen dialect for compiled deployments).
* Additional optimizers: **OpenEvolve**, **ShinkaEvolve**.
* More dialects for specialized targets and runtimes.

---

## Examples in this repo

* `examples/edsl_cot_partitioned.py` — chain‑of‑thought with partitioned reasoning
* `examples/edsl_loop_best_of_n.py` — best‑of‑N with a judge
* `examples/edsl_loop_refine.py` — bounded refine loop with early stop
* `examples/text_ir_showcase.py` — render a prompt as PLIR‑TXT
* `examples/with_sections.py`, `examples/with_input.py`, `examples/minimal_prompt.py`

---

## Project layout

```
pirborg/
  edsl/        # Python EDSL (program, inputs, sections, slots, loops)
  text/        # PIR‑TXT parser, printer, CLI
  optimizer/   # backends (gepa, dspy), registry, runners, scorers
  pir.py       # Prompt IR ops + lowering from PromptSpec
  lints.py     # static lints
  renderer.py  # template rendering helpers
  schema.py    # output JSON Schema builder
examples/      # runnable examples
```

---


## License

See `LICENSE` in the repository.
