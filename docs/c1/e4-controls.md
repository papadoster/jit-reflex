# C1-E4, контрольные прогоны: PPC с латчем и классический регулятор — записка

*Aleksandr Karpov · 2026-10-20 · Статус: **до данных**. Этот текст записан и запушен до пробы и сетки; результаты добавятся после пода отдельным коммитом · [Спек](../superpowers/specs/2026-10-20-c1-e4-controls-design.md) (правила §8, мощность §9, журнал)*

## Семейство: один метод, три способа задать коэффициент

- **G** — геометрический сервопривод с фиксированным коэффициентом κ = 1.0.
- **VS(k)** — он же при другом фиксированном k.
- **G-auto** — он же, где фиксированный коэффициент заменён законом З1.

VS(1.0) и G — один метод, в таблицах он называется G. Вклад G-auto поэтому ровно один: как получить коэффициент. VS(k) подбирает его перебором с эпизодами на каждого кандидата, G-auto измеряет его теневым вызовом на тех же 60 калибровочных случаях.

Для статьи:

> G, VS(k) and G-auto are one geometric servo with three ways to set its gain: G fixes it at 1.0, VS(k) at k, and G-auto replaces the fixed gain with our self-tuning law (Z1). VS(1.0) is G. The only difference G-auto contributes is how the gain is obtained: VS(k) tunes it by grid search, spending episodes on every candidate; G-auto measures it with a shadow call on the same 60 calibration cases.

## Абзацы для статьи

Оба решения приняты до данных и запушены вместе со спеком. Отказ от удобного порога и отказ от удобного конкурента делают предрегистрацию аргументом, а не формальностью.

**1. Почему у сравнения с базовыми методами нет порога** (расчёт мощности — спек §9).

> **Why the baseline comparisons carry no threshold.** We report the differences between our reflex and the two baselines — the fixed-gain servo VS(k) and PPC with its latch — as measurements: an estimate with a 95% cluster-bootstrap interval, and no pass/fail threshold. We fixed this before collecting any data. The reason is power. With 300 paired cases the standard error of a success difference is about 2.7 points, so a non-inferiority rule with a −2-point margin passes only when the point estimate is at least +2.4. At true equality such a rule passes 18% of the time; on the 520 bench pairs, 27%; even if we extended the comparison to all 940 LIBERO-MAX cases, 37%, the power at which our own earlier non-inferiority rule against the fixed-gain servo failed. A rule like this is a superiority test under a non-inferiority label, and at true equality it would report a failure in 63–82% of runs. We reserve hypothesis tests for questions whose outcome can be a finding: whether a gain tuned on one policy transfers to another, whether delay separates the self-tuned and fixed-gain servos, whether the self-tuning law matters under noisy perception, and whether the latch changes PPC where it engages.

**2. Почему мы не ослабляли базовые методы — ни PPC, ни VS.**

> **Why we did not weaken the baselines.** We strengthened both baselines rather than stripping them. To PPC we added the authors' own two-EMA latch stabiliser, which our earlier implementation lacked, as a separate arm with their constants. Where the paper's text (a single trigger holds the latch for two chunks) disagrees with its formulas (four chunks), we implemented the formulas, the reading that favours PPC: the execution horizon stays short for longer, and PPC replans more often. We list the deviations that remain. From our own servo we removed exactly one component, the self-tuning law, to obtain the classical controller VS(k). It keeps the dead band (set by the same rule as ours: at most 5% false engagements on a still object), the per-step limit, the clipping and the stop at the close command. Any servo has a dead band, and no controller keeps correcting towards the object after the gripper closes. We tuned its gain by grid search on the same 60 calibration cases from which our method measures its gain, over a grid {0.1, 0.3, 1.0} that contains values close to the gains our measurement arrived at (0.31 and 0.12). A deliberately weaker baseline would invite exactly the criticism that made us fix PPC. How our method differs from visual servoing is answered by whether a tuned gain transfers across policies and by noise, not by a handicapped competitor.

**3. Правило «всё или ничего» в пробе побитности** (владелец: «пойдёт в статью отдельной строкой»).

> Before the run we repeated pieces of the earlier run on the new machine and compared the hash of every episode's actions. For π0.5 on LIBERO-MAX, 88 of 96 episodes matched bit for bit, as did all 12 event times. The rule fixed in advance was all or nothing: we reused the earlier comparison arms only if every hash matched. We applied it, so for this cell every comparison arm was re-run on the new machine, at a cost of about $2.

**4. Ветка VS остановлена; настоящий регулятор с обратной связью** (владелец: «сильная строка, а не слабая»).

> During the run we found that the branch we had declared as a visual-servoing baseline was, in the code, our own method with a fixed parameter: it fixed the share of the object's shift inherited by the next plan, and it never measured the gripper. We stopped it before reading any of its evaluation results and ran a real closed-loop regulator instead: position-based visual servoing with the reference trajectory given by the policy's own plan.

**5. Место нашего метода в классической теории** (Chaumette & Hutchinson, Part II, IEEE RAM 2007, «Target tracking», ур. 10–12).

> For a moving target, classical visual servoing uses v = −λ L̂⁺ e − L̂⁺ ∂ê/∂t: a feedback term on the measured gripper-to-target error and a feedforward term that cancels the target's own motion. Our reflex is the second term in integrated form (the accumulated object displacement rather than its velocity), with the reference set by the policy's plan; it never measures the gripper. With the plan as the reference and exact execution, a position-based servo at unit gain reduces to our reflex; the two differ only where the arm deviates from the plan for reasons other than object motion. The run measures what the feedback term adds.

Источники: Chaumette & Hutchinson 2006 (Part I, ур. 1, 4) и 2007 (Part II, ур. 10–12); grasp-VS — GG-CNN (Morrison, Corke & Leitner, RSS 2018), Haviland, Dayoub & Corke 2020, Vahrenkamp et al. 2008; честный подбор усиления у регулятора-baseline — Lee, Levine & Abbeel, ICLR 2017 (§6.4, прил. C.5). В статье PPC (2605.11459v2) VS-baseline нет. Спек регулятора: [`2026-10-20-c1-e4-pbvs-design.md`](../superpowers/specs/2026-10-20-c1-e4-pbvs-design.md).

## Результаты

Будут после пода (`results/c1-e4/controls/pod_C/`). Правила C2, C3, C5, C6 и измерения И1–И4 — по спеку §8.
