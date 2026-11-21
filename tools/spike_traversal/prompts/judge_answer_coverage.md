Answer coverage: does the wiki answer the planted questions?

Given the summaries one variant produced and one planted fact (a reader
question plus the reference answer), decide whether the summaries contain
the answer. Judge the meaning, not the wording: a paraphrase that carries
the same information counts as answered, a summary that mentions the topic
without the specific information does not.

You receive:

READER QUESTION:
$question

REFERENCE ANSWER:
$expected_answer

VARIANT SUMMARIES:
$summaries

Decide:

- answered: true when the summaries state the information of the reference
  answer, so a reader of the summaries alone would learn it.
- evidence: the shortest span of the summaries that carries the answer, or
  the empty string when nothing in the summaries answers the question.

Do not use knowledge from outside the summaries. When the summaries only
mention the topic of the question without the specific information of the
reference answer, the fact is not answered.
