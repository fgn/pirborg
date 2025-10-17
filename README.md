<p align="center">
  <img align="center" src="docs/docs/static/img/pirborg_logo.png" width="460px" />
</p>
<p align="left">

# pirborg — Prompt Language Intermediate Representation

> **pirborg is an Intermediate Representation (IR), not a hand-authored prompt language.**  
> You typically **author in a frontend** (e.g., **DSPy** or pirborg’s own authoring frontend), then **lower to PIR** for analysis, linting, optimization, and codegen. The IR is **verbose by design**, machine-friendly, **deterministic**, and **inspectable** (via **PIR‑TXT**). It comes with **template lints** and **textual feedback** so *reflective optimizers* can **self‑adjust** their template layouts during training.

If **DSPy** introduced structured programming to prompt authoring by turning monolithic prompt blobs into modular functions with typed interfaces, then **pirborg** aims to be the MLIR-equivalent for prompts. It sits between high-level authoring frontends (such as DSPy or pirborg’s own EDSL) and multiple backend optimizers, providing a stable intermediate representation (IR) featuring a clear symbol table, explicit control flow, and well-defined operators. This design transforms an M×N problem (many authoring frontends multiplied by many optimizer backends) into an M+N solution by enabling any frontend to target PIR and any backend to optimize from PIR.

DSPy is brilliant, and in most cases, you should continue to use DSPy directly. **pirborg** is designed specifically for advanced compilation workflows, fine-grained optimization, and tooling scenarios where deterministic transformations and robust intermediate representations are essential.


---

## The fast path (first screenful): DSPy → PIR → Optimizer

### 1) Author ergonomically (DSPy)

```python
# Developer-friendly authoring in DSPy
import dspy

class QA(dspy.Signature):
    """Answer precisely, return an 'answer' string."""
    question: str
    answer: str

class Answer(dspy.Module):
    def __init__(self):
        self.pred = dspy.Predict(QA)

    def forward(self, question: str):
        return self.pred(question=question)
```

### 2) Lower to pirborg (IR): same contract, structured internals

> In practice a tool/adapter does this for you; shown here for transparency.

```python
from pirborg.edsl import program, inp, section, output, predict

with program("readme.layers.qa_v1"):
    inp.str("question", channel="user", required=True)

    # Frozen response contract (keeps the optimizer honest)
    section(
        "format",
        channel="system",
        text="Return JSON with an `answer` field only.",
        output=output("answer", description="Primary answer text"),
        description="Response contract.",
    )

    # Thaw persona/style only: optimizer may rewrite this
    section(
        "persona",
        channel="system",
        text="You are concise and accurate. If uncertain, say so.",
        optimizable=True,
        description="Assistant stance and tone.",
    )

    qa_prompt = predict(
        system="{{ emit_section('persona') }}\\n{{ emit_section('format') }}",
        user="Q: {{ inputs.question }}",
        id="readme.layers.qa_v1.base",
    )
```

### 2.5) Equvialent PIR-TXT representation of the above.
The above program can also be represented in **PIR‑TXT**, pirborg’s canonical textual form of the prompt intermediate representation.
It is fully equivalent to pirborg’s Python object representation (**PromptSpec**), and round-trip conversions between them are lossless.
Because it is human readable, and thus also easily interpreted by LLMs, **PIR‑TXT** is ideal for reflective optimizers that inspect, reason about, and modify prompt structures safely and deterministically.

```
pirborg.module @readme.layers.qa_v1 {
  pirborg.input %question : str {channel = "user", required}

  pirborg.section @format {
    channel = "system"
    description = "Response contract."
    text = "Return JSON with an `answer` field only."
    output = {
      key = "answer",
      json_type = "string",
      required = true,
      description = "Primary answer text"
    }
  }

  pirborg.section @persona {
    channel = "system"
    description = "Assistant stance and tone."
    text = "You are concise and accurate. If uncertain, say so."
    optimizable
  }

  pirborg.message "system" {
    emit.section @persona
    emit.literal "\n"
    emit.section @format
  }

  pirborg.message "user" {
    emit.literal "Q: "
    emit.input %question
  }

  pirborg.render { engine = "jinja2" strict_undefined }
}
```

### 3) Optimize (respecting frozen vs. optimizable parts)

```python
from pirborg.optimizer import optimize_prompt, EvalSuite
from pirborg.optimizer.runners import CallableRunner
from pirborg.optimizer.scorers import ContainsSubstringScorer

runner = CallableRunner(lambda messages, instance, candidate: "some model output")
scorer = ContainsSubstringScorer(substr="concise")
evals = EvalSuite(runner=runner, scorer=scorer)

result = optimize_prompt(
    prompt=qa_prompt,       # a PromptSpec
    optimizer="gepa",       # example backend
    trainset=[{"inputs": {"question": "Why is caching fast?"}, "label": "..."}],
    valset=None,
    evals=evals,
)
optimized = result.optimized_prompt
```

**Takeaway:** Author in **DSPy** (or another frontend). Lower to **PIR**. Let the optimizer explore **only** the parts you mark `optimizable=True`, while **frozen** sections (e.g., format contracts, routing) stay invariant.

---

## Ambition

* Make writing **prompt optimizers** simple and safe.
* Provide a **stable IR** that tools can target and generate from.
* Offer **fine‑grained control** over what is optimizable (sections, inputs, slots, routes).
* Support **multi‑call prompt graphs** (e.g., ReAct chains), **bounded loops**, and **selection/judging** patterns.
* Enable **extensible dialects** for code generation (back to DSPy, or to other runtimes such as BAML).
* Ship a **textual format (PIR‑TXT)** and a **deterministic Python EDSL** with round‑trips between them.
* Supply **static analysis** and **lints** for template correctness and schema hygiene.

---

## Highlights

* **IR, not runtime:** A compact Prompt IR (PIR) with a textual form (**PIR‑TXT v1**) and a Python EDSL.
* **Deterministic authoring:** EDSL primitives ensure reproducible structures; templates default to strict semantics.
* **Factor + freeze/thaw:** Mark specific **sections/inputs/routes** as optimizable; freeze the rest to bound the search.
* **Loops & graphs:** Express bounded loops and prompt graphs with selection criteria and optional judges.
* **Optimizers:** Pluggable backends. **GEPA** works today; **DSPy** optimizers MIPROv2, SIMBA etc, **OpenEvolve** and **ShinkaEvolve** on the roadmap.
* **Static checks & feedback:** Lints for unused sections, channel violations, incomplete switches, schema collisions, and more—with **textual feedback** designed for reflective optimizers.
* **CLI tooling:** `pirborg fmt | lint | diff | render` for **PIR‑TXT** files.

---

## Status

Early-stage software, APIs subject to change. Currently in the process of being open-sourced. Feedback is encouraged.

## Install (from source)

```bash
# from a clone of this repo
uv pip install -e .

# optional: optimizer backends
uv pip install gepa     # for GEPA backend
```
> GEPA backend requires the `gepa` package; install as needed.

---

## Blob → Blocks: a worked example (extended)

We start with **one blob**. We end with **factored** sections that an optimizer can safely explore—while our **I/O contract** stays fixed.

### 0) The blob (what you probably ship today)

```python
SYSTEM = "You are a helpful assistant. Return JSON with an `answer` only."
USER   = "Question: {{question}}"
```

### 1) Make inputs/outputs explicit (still one message)

```python
from pirborg.edsl import program, inp, section, output, predict

with program("readme.blob_to_blocks.v1"):
    inp.str("question", channel="user", required=True)

    section(
        "format",
        channel="system",
        text="Return JSON with an `answer` field only.",
        output=output("answer", description="Primary answer text"),
        description="Response contract.",
    )

    prompt_v1 = predict(
        system="{{ emit_section('format') }}",
        user="Q: {{ inputs.question }}",
        id="readme.blob_to_blocks.v1",
    )
```

### 2) Factor into **sections** (start of structure)
We now factor the prompt into named sections. This refactor is shown manually, but a downstream optimizer MAY perform the same semantics‑preserving transformation by applying section.split, which carves out contiguous spans into new sections and replaces them with emit.section at the original call sites—without altering the frozen format contract.

```python
with program("readme.blob_to_blocks.v2"):
    inp.str("question", channel="user", required=True)

    section(
        "persona",
        channel="system",
        optimizable=True,  # let optimizers rewrite only this wording
        text="You are concise, accurate, and cite facts when uncertain.",
        description="Assistant tone and stance.",
    )

    section(
        "format",
        channel="system",
        text="Return JSON with an `answer` field only.",
        output=output("answer", description="Primary answer text"),
        description="Response contract (frozen).",
    )

    base_v2 = predict(
        system="{{ emit_section('persona') }}\\n{{ emit_section('format') }}",
        user="Q: {{ inputs.question }}",
        id="readme.blob_to_blocks.v2.base",
    )
```



### 3) Add **routes** (audience-aware style)

```python
with program("readme.blob_to_blocks.v3"):
    inp.str("question", channel="user", required=True)
    inp.str("audience", channel="user", required=True)  # e.g., "exec" | "student"

    section("persona", channel="system", optimizable=True,
            text="You are concise, accurate, and cite facts when uncertain.")

    section("style_exec", channel="system", optimizable=True,
            text="Use terse bullet points and ROI framing.")

    section("style_student", channel="system", optimizable=True,
            text="Use clear examples and define terms.")

    section("format", channel="system",
            text="Return JSON with an `answer` field only.",
            output=output("answer", description="Primary answer text"))

    base_v3 = predict(
        system=(
            "{{ emit_section('persona') }}\\n"
            "{% if inputs.audience == 'exec' %}{{ emit_section('style_exec') }}"
            "{% else %}{{ emit_section('style_student') }}{% endif %}\\n"
            "{{ emit_section('format') }}"
        ),
        user="Q: {{ inputs.question }}",
        id="readme.blob_to_blocks.v3.base",
    )
```

### 4) Bounded best‑of‑N with a judge (PromptGraph)

```python
from pirborg.edsl import loop_unroll, predict

judge = predict(
    system="Score the candidate 0..1 for factuality and directness.",
    user="Q: {{ inputs.question }}\\nCandidate: {{ inputs.candidate }}",
    id="readme.blob_to_blocks.v4.judge",
)

bo3 = loop_unroll(
    name="best_of_3",
    max_iters=3,
    body=base_v3,
    feed={"question": "inputs.question", "audience": "inputs.audience"},
    select={
        "kind": "argmax",
        "score_field": "score",
        "judge": judge,
        "judge_feed": {"candidate": "step.output.answer"},
    },
)
```

---

## Textual IR vs EDSL (side‑by‑side)

The **same prompt** in PIR‑TXT and the Python EDSL.

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
    emit.literal "\\n"
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
        system="{{ emit_section('persona') }}\\n{{ emit_section('format') }}",
        user="{{ inputs.question }}",
        id="examples.with_sections",
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

runner = CallableRunner(lambda messages, instance, candidate: "some model output")
scorer = ContainsSubstringScorer()
evals = EvalSuite(runner=runner, scorer=scorer)

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

Section(name="closing", text="Summarize...", optimizable=True, description="Closing checklist")

Input(name="segment", type=InputType.ENUM, enum=["new_user","power_user"], optimizable=True,
      optimizer_hints={"default": "new_user"})

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

rendered = render_template(prompt, prompt.template.system_tmpl,
                           inputs={"language": "Swedish", "extraneous": 1},
                           enforce_unknown_inputs=True)  # raises on "extraneous"

issues = lint_prompt(prompt)
for issue in issues:
    print(issue.code, issue.message)
```

This enables optimizers to receive precise and actionable textual feedback, including exact span references and symbol information, when templates violate contracts (e.g., missing placeholders), helping reflective prompt optimizers such as GEPA to make corrections safely and deterministically.

---

## Prompt graphs and tools

Connect multiple prompts and external tools with typed contracts:

```python
from pirborg import PromptSpec, GraphNode, PromptGraph, EdgeBinding, ExternalNode

qa = GraphNode(id="qa", prompt=prompt, entry=True)
grader = GraphNode(id="grader", prompt=judge_prompt)

web = ExternalNode(id="browser", inputs_schema={"query":"string"}, outputs_schema={"snippets":"array"})

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
pirborg diff  left.pirborg right.pirborg      # unified diff
pirborg render path.pirborg --inputs in.json  # render with JSON inputs (and --slots slots.json)
```

---

## Textual IR (PIR‑TXT) round‑tripping

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

* `examples/edsl_blob_to_blocks_v1.py` — one message with input/output  
* `examples/edsl_blob_to_blocks_v2_sections.py` — persona/format split; optimizer scope  
* `examples/edsl_blob_to_blocks_v3_routes.py` — audience‑aware style routing  
* `examples/edsl_blob_to_blocks_v4_bo3.py` — best‑of‑3 with a judge (PromptGraph)  
* `examples/edsl_loop_best_of_n.py` — best‑of‑N with a judge  
* `examples/edsl_loop_refine.py` — bounded refine loop with early stop  
* `examples/text_ir_showcase.py` — render a prompt as PIR‑TXT  
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

## References & further reading

* DSPy: programming with Signatures, modules, and teleprompters — https://dspy.ai/  
* DSPy paper: “DSPy: Compiling Declarative LM Calls into SOTA Pipelines” — https://arxiv.org/abs/2310.03714  
* MLIR textual IR and rationale — https://mlir.llvm.org/docs/LangRef/  
* LLVM IR language reference (text form) — https://llvm.org/docs/LangRef.html 

---

## License

See `LICENSE` in the repository.
