# Task Dependencies

## Terms

- **Dependency** — an arrow between tasks that defines when the next one can start.
- **Lag** — the amount of time that must pass after one task before starting the next.
- **IFS (Interruption-Free Sequence)** — a sequence of consecutive tasks performed without breaks by the same crew.

> **Why it matters (IFS):** It models a continuous block of work, ensuring that tasks performed by the same crew follow
> one another without any interruptions or delays. This is crucial for processes where stopping between steps is
> technologically undesirable or inefficient.

---

## Dependency Types

| Type | How to read                | Meaning (short)                                              |
|------|----------------------------|--------------------------------------------------------------|
| FS   | Finish → Start             | B can start after A finishes (+ lag, if any)                 |
| FFS  | Finish → Start (with lag)  | Same as FS, but with a mandatory pause                       |
| IFS  | Continuous: Finish → Start | B starts immediately after A with no break, by the same crew |
| SS   | Start ↔ Start              | A and B can start together                                   |
| FF   | Finish ↔ Finish            | B must finish no earlier than A                              |

## When to Use Each Type

- Regular sequence of actions: **FS**.
- Technological pause required: **FFS** (or **FS** with a lag).
- Continuous work by the same crew without interruption: **IFS**.
- Simultaneous start of tasks: **SS**.
- Tasks must finish at the same time: **FF**.

---

## Mini-Examples

1) **FS + lag**  
   A: “Pour concrete” (finishes on Day 2) → lag 3 days → B: “Wall masonry” (not earlier than Day 5).

2) **IFS**  
   A: “Drilling” ⇒ B: “Anchoring” ⇒ C: “Post installation” (same crew, no breaks).

3) **SS**  
   A: “Server startup” ≡ B: “Monitoring launch” (simultaneous start).

4) **FF**  
   A: “Main configuration” || B: “Documentation” (must finish at the same time).

5) **FFS**  
   A: “Apply primer” → lag 4 hours → B: “Painting”.

Legend: → (FS/FFS), ⇒ (IFS), ≡ (SS), || (FF).

---

## Example of a CSV File with Dependencies

Suppose we have a file named `predecessors.csv`:

```
child_id,parent_id,type,lag
B,A,FS,3
C,B,IFS,0
D,C,IFS,0
E,A,SS,0
```

Explanation:

- **B** starts after **A** + 3 (lag of 3 days).
- **C** and **D** together with **B** form a continuous chain (**IFS**).
- **E** starts together with **A** (**SS**, does not delay the process).

```