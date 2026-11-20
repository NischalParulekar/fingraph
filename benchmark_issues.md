# Suspected TAT-QA gold-label problems

When a gold answer looks wrong, it is logged here with the evidence. Scores are never
changed: the official scorer's numbers are what the README reports, and this file says
how many failures may be the benchmark's fault rather than the system's.

Found while labelling 150 held-out failures (50 per arm,
`results/heldout/error_labels.csv`). Logged by Claude from the dataset files; not yet
checked by a person. "Strong" means the report itself contradicts the gold answer;
"convention" means the gold follows a reading the question does not ask for.

| Question uid | Split | Gold answer | Problem | Evidence | Strength | Logged |
|---|---|---|---|---|---|---|
| 41a751bd0e1105fa875a7ed4d0051fba | test | 10,705 (million) | Asks for total assets from total liabilities and total equity; gold subtracts (29,810 - 19,105) instead of adding. | The table's own "Total assets" row is 48,915 for 2019, which is liabilities + equity. | strong | 2026-11-20 |
| cf774f6bcde2d8681b99b375c9902ec3 | test | 2011 | Asks when 65-nanometer technology was introduced. | Paragraph 3: "We began our volume production with 65-nanometer and 40-nanometer technologies in 2006 and 2009 ... We introduced our 28-nanometer technology to customers in 2011." | strong | 2026-11-20 |
| d96471e34f661f8d062757ba3f712cd9 | test | 0.76 | Asks for operating cash flow over investing cash flow; gold divides investing by operating (10,085 / 13,189). | Operating 13,189, investing (10,085): the ratio asked for is 1.31. | strong | 2026-11-20 |
| 2870200c1799c44402bc9135e7b68daf | test | "driven by higher returns on plan assets and plan remeasurements; partially offset by • A decrease in net property, plant and equipment of $782 million ..." | Asks what caused the increase in prepaid (pension) assets; the gold span runs on into the next, unrelated bullet. | Paragraph 2 lists "An increase in prepaid pension assets of $2,199 million ... driven by higher returns on plan assets and plan remeasurements; partially offset by • A decrease in net property ..."; the PP&E decrease offsets total noncurrent assets, not prepaid assets. | moderate | 2026-11-20 |
| d06c686c798b7f12bb3217764a542527 | test | 0.29 | Average of 0.47 and 0.12. | (0.47 + 0.12) / 2 = 0.295, which rounds to 0.30 under round-half-up and round-half-even alike. | moderate | 2026-11-20 |
| f005fbea468ed191b98d44c447b868ba | test | "$19.1 million", scale thousand | The answer text names its unit and the scale adds another, so the scorer reads $19.1 thousand-million. | Paragraph 2: "Research and development expense increased by $19.1 million"; the table gives 19,064 (in thousands). Either answer is right; the gold mixes the two. | moderate | 2026-11-20 |
| 3d0d640a8a63cef330ccffb57cf87ceb | test | 2,475,500, scale thousand | Share counts given scale thousand, which makes them 2.5 billion shares. | Paragraph 3 states the counts in full: "The exercisable shares excluded for 2019 and 2018 are 1,216,000 and 1,259,500"; the "in thousands" note belongs to the table. | moderate | 2026-11-20 |
| e798612f7b362bf8def3436937307191 | test | -10 (percent) | Asks for the percentage change from 40% to 30%; gold gives the change in percentage points. | (30 - 40) / 40 * 100 = -25%; the gold derivation is "(30% - 40%) * 100". | convention | 2026-11-20 |
| 8ddab4f696534c10763a19b55229e415 | test | 0.6 (percent) | Asks for the percentage change from 2.7% to 3.3%; gold gives the change in percentage points. | (3.3 - 2.7) / 2.7 * 100 = 22.22%; the gold derivation is "3.3 - 2.7". | convention | 2026-11-20 |

Also seen, not a gold error: the official scorer scores "(3,753)" as 0 against a gold of
-3753, while "-3,753" scores 1 (question e3096b1ac65f83346bcf04e5020d1a3f). An answer
written the accounts way, with parentheses for a negative, is scored wrong.
