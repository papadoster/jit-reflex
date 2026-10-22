# C1-E4, контрольные прогоны: PPC с латчем и классический регулятор — записка

*Aleksandr Karpov · 2026-10-20 · Статус: **результаты прочитаны 2026-10-22** (раздел «Результаты»). Абзацы 1–5 записаны и запушены до пробы и сетки, 6–9 — по ходу прогона, каждый с датой и основанием в журнале; после чтения текст выше раздела «Результаты» не менялся · [Спек](../superpowers/specs/2026-10-20-c1-e4-controls-design.md) (правила §8, мощность §9, журнал)*

## Семейство: один метод, три способа задать коэффициент

- **G** — геометрический сервопривод с фиксированным коэффициентом κ = 1.0.
- **VS(k)** — он же при другом фиксированном k.
- **G-auto** — он же, где фиксированный коэффициент заменён законом З1.

VS(1.0) и G — один метод, в таблицах он называется G. Вклад G-auto поэтому ровно один: как получить коэффициент. VS(k) подбирает его перебором с эпизодами на каждого кандидата, G-auto измеряет его теневым вызовом на тех же 60 калибровочных случаях.

Для статьи:

> G, VS(k) and G-auto are one geometric servo with three ways to set its gain: G fixes it at 1.0, VS(k) at k, and G-auto replaces the fixed gain with our self-tuning law (Z1). VS(1.0) is G. The only difference G-auto contributes is how the gain is obtained: VS(k) tunes it by grid search, spending episodes on every candidate; G-auto measures it with a shadow call on the same 60 calibration cases.

## Абзацы для статьи

Оба решения приняты до данных и запушены вместе со спеком. Отказ от удобного порога и отказ от удобного конкурента делают предрегистрацию аргументом, а не формальностью.

**Абзацы 1 и 2 обязательны в тексте статьи** (владелец, 2026-10-20). В спеке их рецензент не увидит.

**1. Почему у сравнения с базовыми методами нет порога** (расчёт мощности — спек §9).

> **Why the baseline comparisons carry no threshold.** We report the differences between our reflex and the two baselines — PPC with its latch and a position-based visual servo — as measurements: an estimate with a 95% cluster-bootstrap interval. No threshold is assigned, because the sample does not permit one (preregistration §9); we did not silently drop a rule. We fixed this before collecting any data. The reason is power. With 300 paired cases the standard error of a success difference is about 2.7 points, so a non-inferiority rule with a −2-point margin passes only when the point estimate is at least +2.4. At true equality such a rule passes 18% of the time; on the 520 bench pairs, 27%; even if we extended the comparison to all 940 LIBERO-MAX cases, 37%, the power at which our own earlier non-inferiority rule against a fixed-gain servo failed. A rule like this is a superiority test under a non-inferiority label, and at true equality it would report a failure in 63–82% of runs. We keep a hypothesis test only where the outcome can be a finding at this sample size: whether the latch changes PPC where it engages.

**2. Почему мы не ослабляли базовые методы — ни PPC, ни регулятор.**

> **Why we did not weaken the baselines.** We strengthened both baselines rather than stripping them. To PPC we added the authors' own two-EMA latch stabiliser, which our earlier implementation lacked, as a separate arm with their constants. Where the paper's text (a single trigger holds the latch for two chunks) disagrees with its formulas (four chunks), we implemented the formulas, the reading that favours PPC: the execution horizon stays short for longer, and PPC replans more often. We list the deviations that remain. The visual servo closes the loop on the measured gripper position, with the reference given by the policy's own plan. It keeps the two parts every servo has: a dead band and the stop at the close command. They are standard parts of a servo, not our tuning, so we did not remove them. Its dead band is set by the rule that sets our own thresholds, at most 5% false engagements on a still object, and is never below our 5 mm. If that rule gives more than 15 mm, we also run the servo at our own 5 mm gate, so that the competitor is not compared at a gate several times wider than ours. No controller keeps correcting towards the object after the gripper closes. We tuned its gain by grid search over {0.1, 0.3, 1.0} on the same 60 calibration cases from which our method measures its gain. A deliberately weaker baseline would invite exactly the criticism that made us fix PPC.

**3. Правило «всё или ничего» в пробе побитности** (владелец: «пойдёт в статью отдельной строкой»).

> Before the run we repeated pieces of the earlier run on the new machine and compared the hash of every episode's actions. For π0.5 on LIBERO-MAX, 88 of 96 episodes matched bit for bit, as did all 12 event times. The rule fixed in advance was all or nothing: we reused the earlier comparison arms only if every hash matched. We applied it, so for this cell every comparison arm was re-run on the new machine, at a cost of about $2.

**4. Ветка VS остановлена; настоящий регулятор с обратной связью** (владелец: «сильная строка, а не слабая»).

> During the run we found that the branch we had declared as a visual-servoing baseline was, in the code, our own method with a fixed parameter: it fixed the share of the object's shift inherited by the next plan, and it never measured the gripper. We stopped it before reading any of its evaluation results and ran a real closed-loop regulator instead: position-based visual servoing with the reference trajectory given by the policy's own plan.

**5. Место нашего метода в классической теории** (Chaumette & Hutchinson, Part II, IEEE RAM 2007, «Target tracking», ур. 10–12).

> For a moving target, classical visual servoing uses v = −λ L̂⁺ e − L̂⁺ ∂ê/∂t: a feedback term on the measured gripper-to-target error and a feedforward term that cancels the target's own motion. Our reflex is the second term in integrated form (the accumulated object displacement rather than its velocity), with the reference set by the policy's plan; it never measures the gripper. With the plan as the reference and exact execution, a position-based servo at unit gain reduces to our reflex; the two differ only where the arm deviates from the plan for reasons other than object motion. The run measures what the feedback term adds.

**6. Ошибочно помеченная ветка и привилегированный вариант, который мы не гоняли** (владелец: «сказать прямым текстом»).

> The 108 completed cases of the stopped branch, labelled VS(1.0), are in the code exactly our servo G at gain 1.0: the same dead band, step limit, clipping and stop at the close command, and no gain law; a bitwise test confirms it. We use them under their correct name, G(k = 1.0), as a paired reference for the position-based servo, and we recorded this decision before reading any of their outcomes. We did not run a teach-by-showing variant whose reference is the gripper-to-object offset at the grasp in the paired unperturbed episode. It is stronger by construction but privileged: it needs an episode that never happened, which a real robot cannot have, whereas an oracle object tracker stands in for a perfect camera.

**7. Teach-by-showing: потолок классического регулятора и честная оговорка** (владелец, 2026-10-21).

> The plan-referenced servo needs a model of the arm, and its dead band (26 mm here) is that model's error, a property of the variant we chose rather than of visual servoing. We therefore also ran the classical teach-by-showing servo, whose reference is a gripper-to-object offset and which needs no arm model; its dead band is our own 5 mm. Its reference s* is the offset at the grasp in the paired unperturbed episode, an episode that never happened in the perturbed run, which is knowledge a real robot does not have. A deployable teach-by-showing servo would take s* from a demonstration recorded in advance, and LIBERO ships demonstrations for its tasks, so this privilege is a property of our implementation, not of the approach. We did not run the demonstration variant: our cases are perturbed LIBERO-Plus and LIBERO-PRO scenes, so a demonstration's grasp would have to be matched to the object and would itself be an approximation, whereas our reference is the policy's own grasp in the very same scene. It is the natural ceiling of this baseline, though not a strict one, since a demonstration may grasp better than the policy did; we report how often the reference grasp itself succeeded. If the teach-by-showing servo is ahead of our reflex, the contribution reads: our reflex reaches its result without this knowledge.

**8. Покрытие классов помех — сужено самими** (владелец, 2026-10-22: «до того, как спросят»; в раздел ограничений отдельным пунктом).

> All four perturbation classes we test are translations of a single object under different dynamics: a jump, smooth motion, noisy eyes and inference delay. We did not test rotation of the object, a change of the grasp point, contact events or several moving objects, and our correction is blind to them by construction: it adds a translation to the plan, not a rotation.

**9. Наследованный λ у ветки VT-m — смещение в нашу пользу, раскрыто** (владелец, 2026-10-22; спек PBVS, журнал «ветка VT-m»).

> The teach-by-showing servo with memory (VT-m) inherits its gain λ = 0.3 and its 5 mm dead band from the plan-referenced and the teach-by-showing servos, without a selection of its own. Its error has a term the other two lack, the plan's predicted aim relative to the demonstration, so a gain chosen for their errors may be suboptimal for it. We did not tune it on purpose: by then we had seen the summaries, and any selection would have been post hoc. A suboptimal gain weakens exactly the branch whose win would overturn our claim, so this bias favours our conclusion. If VT-m still beats our reflex, the conclusion against us is the stronger for it; if it does not, part of the gap may be the price of the inherited gain.

Источники: Chaumette & Hutchinson 2006 (Part I, ур. 1, 4) и 2007 (Part II, ур. 10–12); grasp-VS — GG-CNN (Morrison, Corke & Leitner, RSS 2018), Haviland, Dayoub & Corke 2020, Vahrenkamp et al. 2008; честный подбор усиления у регулятора-baseline — Lee, Levine & Abbeel, ICLR 2017 (§6.4, прил. C.5). В статье PPC (2605.11459v2) VS-baseline нет. Спек регулятора: [`2026-10-20-c1-e4-pbvs-design.md`](../superpowers/specs/2026-10-20-c1-e4-pbvs-design.md).

## Результаты

*Прочитано 2026-10-22, после конца пода C. Данные: `results/c1-e4/controls/pod_C/` (`grid_all_{C,V,T,K,M}.jsonl`, MANIFEST 171/171). Чтения на Маке: `results/c1-e4/controls/summary_{C,V,T,K,M}.json` (`scripts/c1e4c_summary.py`, `c1e4v_`, `c1e4t_`, `c1e4k_`, `c1e4b_`; код чтений запушен до данных, кроме A — метод записан постфактум, — и D — код до вычисления, но после данных VS-T). Под C: RTX 4090, π0.5 `lerobot/pi05_libero_finetuned_v044` @ `8e174154`, LIBERO-MAX, первые 300 случаев оценки (187 Plus + 113 PRO), если не сказано иное.*

**Как читать числа:**
- **Разности** — успех в п.п., парно по случаю и seed'у. В скобках — 95% бутстрэп с кластерами по исходной задаче (33 в LIBERO-MAX, 26 на стенде), 2000 пересборок. У C и D — 8000 пересборок, все границы из них. У A из 8000 только односторонняя верхняя граница, двусторонний интервал — из 2000.
- «Одност. нижняя» / «одност. верхняя» — граница односторонних 95%.
- «Через под» — сторона из части 2 (другой под). У π на LIBERO-MAX режим С, такие пары по случаю, но не побитные.

### Итог

| что | разность | 95% | одност. | чтение по правилу |
|---|---|---|---|---|
| **C5:** PPCL − PPC, пары с латчем (91 пара, 31 задача) | +1.1 | −5.6…+7.0 | — | **не меняет различимо** → ЛАТЧ МЕНЯЕТ PPC: **НЕТ** |
| **И4:** G-auto − PPCL, стенд, оракул, 520 пар | **+7.1** | +2.1…+12.3 | нижн. +2.9 | измерение |
| **И5:** PPCL@glr − PPC@glr, M1, 300 пар | −2.0 | −4.9…+1.0 | — | измерение |
| **Н1:** PBVS − G-auto, M1, d = 0 | +2.3 | −3.7…+8.0 | — | измерение |
| **Н2:** PBVS − G-auto, d = 10 и 20 в пуле, 597 пар | −2.8 | −7.2…+1.2 | — | измерение |
| **A:** PBVS − G-auto, шесть клеток, 897 пар (метод записан постфактум) | −1.1 | −4.3…+2.1 | **верхн. +1.6** | «член обратной связи PBVS добавляет к G-auto не более 1.6 п.п.» |
| **Т1:** VS-T − G-auto, M1, d = 0 | **+6.4** | +1.0…+11.7 | — | измерение |
| **D, с задержкой:** VS-T − G-auto, d = 10 и 20, 598 пар | −1.3 | −5.0…+2.2 | **верхн. +1.7** | «под задержкой он добавляет не более 1.7 п.п.» |
| **C:** тренд T = D(20) − D(0), 108 случаев Plus | +0.9 | −9.1…+11.4 | нижн. −7.5 | **рост с задержкой не показан** |
| **B, главное:** VT-m − G-auto, d = 0 PRO, 113 пар | **+12.4** | +4.2…+20.5 | нижн. **+5.6** | **исход (i): «плоская кривая» снимается** |
| **B, главное:** VT-m − G-auto, d = 10 Plus, 186 пар | −4.3 | −10.2…+1.2 | верхн. +0.5 | (исход по обеим клеткам — (i)) |
| **В1:** VT-m − VS-T, d = 10 Plus | +2.1 | −5.3…+9.0 | нижн. −4.0 | **не показано** |

Каждое чтение — отдельное утверждение, поправки на множественность нет.

- **Латч не меняет PPC различимо** (C5). На 91 паре M1, D и стенда, где латч держал хоть один чанк, разность +1.1 [−5.6; +7.0]. По клеткам целиком PPCL − PPC: +0.3 / +0.3 / 0.0 при d = 0 / 10 / 20, −0.8 и 0.0 на стенде A и A20.
- **И4, плавное движение:** G-auto − PPCL +7.1 [+2.1; +12.3] на стенде, интервал без нуля. Это измерение без порога, слов «лучше» и «не хуже» оно не даёт. На LIBERO-MAX, только отчёт: G-auto − PPCL +3.0 [−2.0; +8.0] при d = 0, **+11.3** [+5.4; +17.7] при d = 10, **+7.7** [+2.7; +12.7] при d = 20.
- **Шумные глаза, по толкованию, записанному до прогона клетки:** PPCL@glr наш рефлекс не обогнал. PPCL@glr − G@glr (через под) −8.0 [−12.1; −4.3], при этом у PPCL@glr в среднем на 4.7 вызова мозга за эпизод больше (32.8 против 28.1, медиана +2). Вывод части 2 о шумных глазах стоит и с починенным конкурентом.
- **Классический PBVS без привилегий: различие с G-auto не показано ни в одной клетке.** Это не доказывает равенства. Объединённая граница: член обратной связи добавляет к G-auto не более 1.6 п.п. (A). PBVS − PPCL — во всех трёх клетках интервал без нуля (Н3, ниже).
- **Привилегированный teach-by-showing** (две фразы записи D): без задержки он впереди на 6.4 п.п. [+1.0; +11.7]; под задержкой добавляет не более 1.7 п.п.
- **Закон κ:** рост его преимущества с задержкой не показан (C). Устойчивость под задержкой закону κ в статье не приписывается. Формулировка сужается до измеренных D(d) (раздел C ниже).
- **VT-m (цель из показа с памятью):** при d = 0 PRO VT-m − G-auto +12.4, одност. нижняя +5.6 > 0. По правилу это исход (i): **«плоская кривая» снимается. Вывод: «информация полезна, беспамятная реализация неудачна». Пишем так, даже если это рушит тезис сводки.** Что за этим стоит по числам — раздел B ниже.

### Проверки перед чтением

Все пять чтений: **PASS**.
- **Полнота:** ни одной недостающей записи ни в одной строке. Строки контрольного прогона: 1 — 240, 2 — 1008, 3 — 1800, 7 — 520, проход C5 — 524. Строки 4, 5, 6 и 8 сняты до данных (журнал контрольного спека, «ветки VS остановлены»). Каждая строка бегуна кончилась «failed 0», второго прохода не было ни разу.
- **Совпадение e** нового Base с Base линии-опоры: 298–300 из 300 в каждой клетке; в B — 113/113 и 187/187.
- **Негодные пары** ≤ 2% везде. Наибольшее — 2 из 300 (Н2/d20, Т1/d0, Н3/d20), в C — 1 из 187 (d = 20).
- **Подбор:** k_π = VS1.0 (промах 150.8 мм против 173.2 и 183.5, 50 годных из 60); λ PBVS = 0.3 (216.66 / 216.77 / 217.23 мм, 58 из 60); λ VS-T = 0.3 (134.5 / 139.2 / 135.2 мм, 57 из 60). Сводка пересчитала все три и сверила метки в записях.
- **Пробы побитности** (§7.1): π LIBERO-MAX 88/96 → режим С; OFT+ 0/60 → С (OFT+ потом снят вместе с VS); π стенд 80/80 → П.
- **Зона PBVS:** модель A, δ = 26.2 мм (> 15 мм) → строка V2eps на 5 мм прогнана, как записано до данных.
- **B:** конфигурация VT-m — модель `band.json` при δ = 5 мм; s* у всех пар совпадает с парным Base.

### Контрольный прогон: PPC с латчем

**C5 и латч.** Доля эпизодов PPCL с латчем, триггеров и чанков под латчем:

| клетка | эпизоды | с латчем | чанков на эпизод | триггеров на эпизод | сбросов у предмета |
|---|---|---|---|---|---|
| M1, d = 0 | 300 | 7.7% | 0.39 | 0.11 | 0 |
| D, d = 10 | 300 | 5.7% | 0.28 | 0.10 | 0 |
| D, d = 20 | 300 | 2.3% | 0.10 | 0.04 | 0 |
| стенд A | 260 | 13.8% | 0.73 | 0.32 | 0.02 |
| стенд A20 | 260 | 3.1% | 0.07 | 0.04 | 0 |

- **Граница худшего случая** (правило журнала до подсчёта, по клеткам строк 2 и 3):
  - **M1:** B = 7.7 п.п. (6 см 7.0, 12 см 8.3) не меньше разрыва G-auto − PPC части 2 (+2.7, на 12 см +4.7). По правилу это идёт **владельцу, без выводов**. Решения владельца пока нет;
  - **D, d = 10 и 20:** B = 5.7 и 2.3 меньше разрыва части 2 (31.0 − 19.3 ≈ +11.7 и 16.7 − 9.0 ≈ +7.7). Строка по правилу: «латч сработал в 5.7% и 2.3% эпизодов, максимально возможное влияние 5.7 и 2.3 п.п., вывод не меняется».
- **Сам эффект латча измерен проходом C5:** PPCL − PPC на всех 300 случаях M1 +0.3 [−0.7; +1.7]; на 91 залатченной паре M1, D и стенда вместе +1.1 [−5.6; +7.0] (C5); на всём пуле 1420 пар 0.0 [−0.5; +0.4]. Верхняя граница на залатченных парах (+7.0) выше разрыва M1 (+2.7), так что C5 правило границы для M1 не закрывает.
- **OFT+ и OFT:** перенос доли латча с π на OFT-мозги заменён ограничением (журнал, «четыре ноги»). Прогон PPCL на OFT+ или OFT — решение владельца после C5 (≈ $0.8 за клетку 300 случаев).

**Разности на LIBERO-MAX** (π, 300 случаев; G-auto и PPCL на этом поде, «нет» через под):

| d | G-auto − PPCL | PPCL − нет | PPCL − PPC |
|---|---|---|---|
| 0 | +3.0 [−2.0; +8.0] | +4.0 [−0.7; +8.7] | +0.3 [−0.7; +1.7] |
| 10 | **+11.3** [+5.4; +17.7] | **+8.4** [+4.4; +12.1] | +0.3 [−0.7; +1.7] |
| 20 | **+7.7** [+2.7; +12.7] | **+5.0** [+1.0; +9.1] | 0.0 |

- Срезы M1: G-auto − PPCL на 6 см 0.0 [−6.7; +6.1], на 12 см +5.7 [−1.2; +12.9].
- Стенд: PPCL − нет +8.1 [0.0; +16.2] в A, +9.6 [+5.0; +14.2] в A20.
- **Успех, %** (π, 300 случаев): G-auto 49.0 / 31.7 / 16.7 при d = 0 / 10 / 20; PPCL 46.0 / 20.3 / 9.0; «нет» части 2 на тех же 300 случаях 42.0 / 12.0 / 4.0. Стенд: PPCL 58.1 в A, 13.5 в A20.
- **Вызовы мозга на эпизод** (плановые + внеплановые): при d = 0 G-auto 27.6 + 0, PPC 28.1 + 1.0, PPCL 28.0 + 1.3, «нет» 29.2 + 0. Рефлекс вызовов не добавляет.

**Шумная клетка** (M1, π, 300 случаев, glr-глаза; таблица записана до прогона клетки):

| ветка | успех, % | вызовов на эпизод (среднее, медиана) | доля эпизодов с латчем |
|---|---|---|---|
| PPCL@glr | 39.3 | 32.8, 30 | 58.7% |
| PPC@glr (этот под) | 41.3 | 30.2, 29 | — |
| G@glr (часть 2, через под) | 47.3 | 28.1, 28 | — |
| нет (часть 2, через под) | 42.0 | 29.2, 28 | — |

- **И5** = PPCL@glr − PPC@glr: −2.0 [−4.9; +1.0].
- **Только отчёт:** PPCL@glr − G@glr −8.0 [−12.1; −4.3] на 299 парах; вызовов у PPCL@glr больше на 4.7 за эпизод (медиана +2).
- **Удержание на шуме** (разность с шумом против разности на оракуле, обе против «нет»): PPCL −2.7 против +4.0; в части 2 G +5.3 против +8.7 (61%), PPC −0.7 против +4.0.

### PBVS: регулятор с обратной связью, эталон из плана

| | d = 0 | d = 10 | d = 20 |
|---|---|---|---|
| **Н1, Н2:** PBVS − G-auto | +2.3 [−3.7; +8.0] | −4.7 [−11.3; +2.0] | −1.0 [−5.6; +3.7] |
| **Н3:** PBVS − PPCL | **+5.3** [+0.3; +10.9] | **+6.7** [+0.7; +13.0] | **+6.7** [+1.7; +11.4] |
| PBVS − нет (через под) | **+9.4** [+3.7; +14.6] | **+15.1** [+9.0; +20.9] | **+11.7** [+7.6; +15.9] |
| успех PBVS, % | 51.3 | 26.7 | 15.7 |

- **Н2 в пуле** d = 10 и 20: −2.8 [−7.2; +1.2], 597 пар.
- **A — объединённая граница по шести клеткам** (d = 0, 10, 20 × Plus и PRO; 897 пар, 33 кластера, 3 негодные): −1.1 [−4.3; +2.1], **односторонняя верхняя +1.55**. Читается, как записано: «член обратной связи PBVS добавляет к G-auto не более 1.6 п.п.». **Метод применён до записи и записан постфактум** (журнал PBVS, 2026-10-22, A). Схема одна, запущена один раз. В статью идёт число кода; число учебника может отличаться в пределах случайности пересборки.
- **Зона 5 мм** (V2eps, d = 0): PBVS − G-auto +3.0 [−2.7; +8.2]; PBVS на 5 мм − PBVS на 26 мм +0.7 [−4.3; +5.7]. Успех 52.0%.
- **При d = 0 мёртвая зона гасит большую часть окна:** при зоне 26 мм 84% шагов окна без поправки (6 см 84%, 12 см 84%), при 5 мм — 52%. Под задержкой при 26 мм — 46% и 49%.
- **PBVS − G(k = 1.0)** на 108 замороженных случаях Plus (только отчёт): +1.9 [−6.1; +10.9].
- **Перелёт** (смена знака проекции e на направление сдвига): 0.14 на эпизод при d = 0 и 26 мм; 1.35 при 5 мм; 1.08 и 0.70 под задержкой.
- **Промах захвата**, PBVS − G-auto: +5.1 / +7.4 / +7.3 мм при d = 0 / 10 / 20.
- **Средняя |e|** в окне: 25.6 мм (5–95%: 3.7…47.7) при d = 0 и 26 мм; 24.3 мм при 5 мм; 64.1 и 62.6 мм при d = 10 и 20.
- **Вызовы мозга на эпизод** (плановые; внеплановых нет): 27.2 / 30.7 / 32.3 при d = 0 / 10 / 20, у G-auto 27.6 / 30.0 / 31.9.
- **Модель руки:** выбрана A, δ = 26.2 мм; все δ_M: I 30.4, A 26.2, I* 28.8, A* 28.1 мм.
- Срезы M1: PBVS − G-auto на 6 см +0.7 [−7.0; +7.9], на 12 см +3.8 [−4.1; +11.6]; PBVS − PPCL на 6 см +0.7 [−6.9; +8.4], на 12 см **+9.6** [+3.9; +15.2].

Н1 согласуется с ожиданием P2 спека PBVS (догадка, не правило; «не хуже» измерение установить не может). Фраза P2, записанная до данных: «наш рефлекс — это упреждающий член классического visual servoing при эталоне, заданном планом политики; прогон измеряет, что добавляет член обратной связи». По A член обратной связи добавляет не более 1.6 п.п.

### VS-T: teach-by-showing, привилегированный потолок

| | d = 0 | d = 10 | d = 20 |
|---|---|---|---|
| **Т1:** VS-T − G-auto | **+6.4** [+1.0; +11.7] | −3.3 [−8.4; +1.7] | +0.7 [−3.4; +4.3] |
| VS-T − PBVS | +4.0 [−1.0; +8.8] | +1.3 [−3.7; +6.4] | +1.7 [−2.7; +6.4] |
| VS-T − PPCL | **+9.4** [+3.7; +14.9] | **+8.0** [+3.4; +12.4] | **+8.4** [+3.0; +13.4] |
| VS-T − нет (через под) | **+13.4** | **+16.4** | **+13.3** |
| успех VS-T, % | 55.3 | 28.0 | 17.3 |
| качество эталона: «закрыть» в парном Base привела к захвату | 88% | 76% | 66% |

- **D — объединённая граница** (до вычисления, журнал PBVS 2026-10-22, D):
  - без задержки (d = 0, 298 пар, 33 кластера, выбыли 2 по e): +6.4 [+1.0; +11.7], односторонняя верхняя +10.9;
  - с задержкой (d = 10 и 20, 598 пар, 33 кластера, выбыли 2 по e): −1.3 [−5.0; +2.2], **односторонняя верхняя +1.7**;
  - вторичная строка без пар, где парный Base не закрылся (s* нет: 9 и 16 пар): без задержки +5.9 [+0.3; +11.4], верхняя +10.5; с задержкой −2.1 [−5.6; +1.3], верхняя +0.8.
- **Две фразы для статьи** (форма из записи D): без задержки привилегированный вариант впереди на 6.4 п.п. [+1.0; +11.7]; под задержкой он добавляет не более 1.7 п.п.
- По прочтению, записанному до данных: «VS-T впереди G-auto» — нормальный исход. Вклад: наш рефлекс без знания захвата из эпизода, которого не было, достигает 49.0% при d = 0 против 55.3% у VS-T, и под задержкой не уступает ему больше чем на 1.7 п.п.
- Эпизоды без эталона: 9 / 8 / 8 из 300. Доля шагов окна с поправкой 86–91%, перелёт 0.9–1.2 на эпизод. Средняя |e| 54.6 / 55.5 / 57.9 мм.
- Промах захвата, VS-T − G-auto: −6.8 / +0.6 / +1.3 мм при d = 0 / 10 / 20.

### C: абляция закона κ по оси задержки

D(d) = G-auto − G(k = 1.0), π, Plus. k = 1.0 выбран правилом §5 до любых оценочных данных.

| | 108 общих случаев | 187 случаев |
|---|---|---|
| D(0) | −0.9 [−10.4; +7.9] | — |
| D(10) | +5.6 [0.0; +11.1] | **+5.3** [+1.0; +10.2] |
| D(20) | 0.0 [−2.7; +2.8] | 0.0 [−2.1; +2.1] (186 пар) |
| **T = D(20) − D(0)** | **+0.9** [−9.1; +11.4], одност. нижняя −7.5 | — |
| D(10) − D(0) | +6.5 [−3.7; +16.7] | — |

- **Чтение:** нижняя граница T не больше нуля → **«рост с задержкой не показан»**. Предсказание против нас (ноль вероятен) сбылось.
- **Что следует для статьи** (записано до данных): устойчивость под задержкой закону κ не приписывается; формулировка сужается до измеренных D(d). При d = 10 на 187 случаях D(10) = +5.3 [+1.0; +10.2], интервал без нуля. При d = 0 и d = 20 различие не показано.
- Успех G(k = 1.0): 50.0% при d = 0 (108 случаев), 27.8% и 17.6% при d = 10 и 20 (187).

### B: VT-m — цель из показа с памятью

| | d = 0 PRO, 113 пар | d = 10 Plus, 186–187 пар |
|---|---|---|
| **главное:** VT-m − G-auto | **+12.4** [+4.2; +20.5], одност. нижняя **+5.6** | −4.3 [−10.2; +1.2], одност. верхняя +0.5 |
| **В1:** VT-m − VS-T (только память) | +1.8 [−4.5; +8.0] | +2.1 [−5.3; +9.0], одност. нижняя −4.0 |
| **В2:** VT-m − PBVS на 5 мм (только цель) | **+8.0** [+2.1; +14.5] | 0.0 [−6.9; +6.8] |
| В2 против PBVS на 26 мм (только отчёт) | — | +1.1 [−6.5; +8.8] |
| успех, %: VT-m / G-auto / VS-T / PBVS 5 мм | 50.4 / 38.1 / 48.7 / 42.5 | 28.3 / 33.2 / 26.2 / 28.3 |

- **Пол шума** (223 эпизода, где определены и пол, и сигнал): медиана |a_R| 16.6 мм, медиана |a_N − a_R| 43.9 мм, отношение 2.64 ≥ 2 → **показательно**. Поэтому В2 измерен в обеих клетках.
- **Главное — исход (i)** (по правилу записи «ветка VT-m»): нижняя односторонняя граница больше нуля в клетке d = 0 PRO. Запись по правилу: **«плоская кривая» снимается. Вывод: «информация полезна, беспамятная реализация неудачна». Пишем так, даже если это рушит тезис сводки.** Статью исход не рушит: VT-m требует захвата из эпизода, которого не было, а у настоящего робота такого знания нет.
- **Что стоит за исходом (i) по числам** (наблюдения после данных, помечено):
  - (i) дала одна клетка, d = 0 PRO. При d = 10 Plus различие VT-m − G-auto не показано: −4.3 [−10.2; +1.2], одност. верхняя +0.5;
  - **В1 не показано** ни в одной клетке. Различие только в памяти (VT-m против VS-T) в этих данных не видно. Вторая половина записанной фразы («беспамятная реализация неудачна») числом В1 не подкреплена — это толкование после данных;
  - **В2 при d = 0** +8.0 [+2.1; +14.5], интервал без нуля; ветки отличаются только целью (показ против плана), модель руки, порог и λ одни. При d = 10 — 0.0 [−6.9; +6.8];
  - клетки различаются не только d, но и источником: при d = 0 только PRO, при d = 10 только Plus.
- **Счётчики:** s* нет в 5 и 4 эпизодах; доля шагов окна с прицелом 71% и 27%; шагов без определённого прицела 5.5% и 2.2%.
- **Наследованные λ = 0.3 и порог 5 мм** — смещение против VT-m, раскрыто (абзац 9). В клетке d = 0 PRO VT-m обогнал G-auto и с этим смещением.

### Что меняется для статьи

- **Остаётся:**
  - против починенного PPC по клеткам: G-auto − PPCL +3.0 [−2.0; +8.0] / +11.3 [+5.4; +17.7] / +7.7 [+2.7; +12.7] при d = 0 / 10 / 20; правила на тренд по d нет (C2 снят до данных);
  - латч не меняет PPC различимо (C5); на шумных глазах PPCL@glr рефлекс не обогнал и потратил больше вызовов (толкование до прогона клетки). Вывод про латч на M1 — за владельцем: граница худшего случая там не меньше разрыва;
  - на плавном движении G-auto − PPCL +7.1 [+2.1; +12.3] (И4, интервал без нуля);
  - член обратной связи PBVS добавляет к G-auto не более 1.6 п.п. (A, с пометкой «метод записан постфактум»).
- **Сужается:**
  - устойчивость под задержкой закону κ не приписывается (C); D(d) с интервалом без нуля только при d = 10 (+5.3 [+1.0; +10.2]);
  - привилегированный teach-by-showing без задержки впереди на 6.4 п.п. [+1.0; +11.7], под задержкой добавляет не более 1.7 п.п. (D).
- **Снимается:** тезис «плоской кривой требований» — что всё сверх смещения предмета ничего не покупает (B, исход (i)). При d = 0 PRO: VT-m − G-auto +12.4 [+4.2; +20.5], VT-m − PBVS с той же моделью +8.0 [+2.1; +14.5].
- **Ограничения, добавленные чтением:**
  - В2 измерен при d = 0 (PRO) и при d = 10 (Plus); при d = 20 не измерялся;
  - VT-m прогнан в двух клетках, а не в шести; В1 не показано ни в одной;
  - доля латча измерена на π, на OFT-мозги не переносится; PPCL на OFT+ и OFT не прогонялся;
  - исполнитель с 2026-10-22 не полностью слеп (журнал PBVS): защитой служат формулы, правила и толкования, записанные до данных C и B.

### Деньги

Под C работал с 2026-10-20 10:55 до 2026-10-22 20:03 UTC (Community RTX 4090, $0.371/ч с диском), все стадии без отказов. После конца сетки остановлен; хранение диска — $0.033/ч. Остаток на счёте $3.30.
