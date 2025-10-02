# Connections between tasks

## Terms

- Connections — an arrow between tasks that indicates when the next one can start.
- Lag — how long you need to wait after one task before starting the next.
- Uninterrupted chain (IFS) — several tasks in a row without a break, performed by the same crew.

> Why this is needed (IFS): to avoid starting the next task too early and to observe technological pauses (drying, cooling, etc.).

---

## Connections types

| Type | How to read               | Short meaning                                           |
|------|---------------------------|---------------------------------------------------------|
| FS   | Finish → Start            | B can start after A finishes (+ lag, if any)           |
| FFS  | Finish → Start (with lag) | Same, but with a mandatory lag                         |
| IFS  | Uninterrupted: Finish → Start | B goes right after A without a break by the same crew |
| SS   | Start ↔ Start             | A and B can start together                             |
| FF   | Finish ↔ Finish           | B must finish no earlier than A                        |

---

## When to use which type

- Ordinary sequence of actions: FS.
- Need to observe a technological pause: FFS (or FS with a lag).
- Need uninterrupted work by the same crew without a break: IFS.
- Common simultaneous start: SS.
- Want to “converge” to a single finish time: FF.

---

## Mini examples

1) FS + lag  
   A: “Pour concrete” (finish Day 2) → lag 3 days → B: “Bricklaying” (no earlier than Day 5).

2) IFS  
   A: “Drilling” ⇒ B: “Anchoring” ⇒ C: “Stand installation” (one crew, without a break).

3) SS  
   A: “Server launch” ≡ B: “Start of monitoring” (simultaneously).

4) FF  
   A: “Primary configuration” || B: “Documentation” (finish by the same deadline).

5) FFS  
   A: “Apply primer” → lag 4 hours → B: “Painting”.

Legend: → (FS/FFS), ⇒ (IFS), ≡ (SS), || (FF).

---

## Example file with dependencies (CSV)

Let this be the file predecessors.csv:

```
child_id,parent_id,type,lag
B,A,FS,3
C,B,IFS,0
D,C,IFS,0
E,A,SS,0
```

What’s here:

- B after A + 3 (lag 3).
- C and D together with B form an uninterrupted chain (IFS).
- E starts together with A (SS, does not delay).