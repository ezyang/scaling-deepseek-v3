# 04 — Choosing the architecture against the hardware

Working outline; unpublished. The asymptotic justification of 6ND stays in 03.

**Premise:** 03 fixes DSv3 and asks which parallelizations can afford their
communication. 04 reverses the question: with hardware and a training budget
given, what architecture should we choose? Focus on expert granularity rather
than attempting to derive every DSv3 hyperparameter.

1. **Why want smaller experts?** Establish the modeling motivation for finer
   experts from primary sources. Treat this as an empirical premise to examine,
   not something the hardware roofline proves.
2. **Define the controlled comparison.** Fix model width and, initially, total
   and activated routed-expert parameters. Narrow experts while increasing
   expert count and top-k proportionally. Expert GEMM work stays approximately
   fixed; routing work and activation communication need not. These are the
   comparison's assumptions, not a universal scaling recipe.
3. **Spend the communication headroom.** Reuse 03's overlap inequality. In the
   simple per-assignment model, expert compute scales with model width × expert
   intermediate width, while communication scales with model width. Narrower
   experts therefore reduce compute per communicated byte. Solve for how narrow
   they can get before communication becomes exposed.
4. **Intersect the constraints.** Bring back 02's memory limits and account for
   GEMM shapes, tokens per expert, routing/node restrictions, and the schedule's
   actual overlap opportunities. Sufficient aggregate compute does not by itself
   guarantee overlap. Show a feasible region, not a uniquely derived optimum.
5. **Locate DSv3 and change the hardware.** Put its design on that picture. What
   changes with faster compute at the same network bandwidth, or a faster
   network at the same compute throughput? Which architectural choices become
   possible, and which still require modeling experiments?

Possible signature widget: expert-width slider, with expert count and top-k
coupled to preserve the chosen parameter budgets; show compute time,
communication time, and the overlap boundary. Start with the explicitly
simplified traffic model; distinguish it from node-aggregated DSv3 traffic.

Source check to retain: DSv3 §3.2.1 describes its approximately 1:1
computation-to-communication ratio as an “inefficient” ratio motivating DualPipe.
That is not evidence by itself that expert width was chosen by minimizing it
subject to overlap. Present this essay as a reconstructed design exercise unless
additional sources establish the actual design history.

[^ai]: AI-drafted planning notes based on the author's discussion; requires a
    human edit pass before use as essay prose.

Draft disclosure.[^ai]
