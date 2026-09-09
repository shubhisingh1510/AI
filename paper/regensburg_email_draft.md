# Draft: data-access request to Dr. Sally Kempa (Universitätsklinikum Regensburg)

Not sent. Review, fill in the bracketed fields, and send yourself from your own email client —
this is a real request to a real researcher, so it should go out under your name, not be sent
by me.

**To:** sally.kempa@ukr.de
**Subject:** Data access request — venous/arterial ulcer image dataset (Neuwieser et al. 2025, *Diagnostics*)

---

Dear Dr. Kempa,

I am [your name], [your affiliation — e.g. "an independent researcher" / "a student at ___"],
working on a research project comparing classical and quantum-hybrid deep learning approaches
for classifying lower-limb ulcers by etiology (venous, arterial, diabetic).

I read your paper "Interpreting Venous and Arterial Ulcer Images Through the Grad-CAM Lens"
(*Diagnostics* 2025, 15(17):2184), and I am writing to ask about access to the venous/arterial
ulcer image dataset described there (198 arterial, 409 venous images, diagnoses confirmed by
Doppler/duplex ultrasound and ankle-brachial pressure index), per the paper's Data Availability
Statement.

My current pipeline uses the public AZH Wound and Vascular Center dataset, which covers venous,
diabetic, pressure, and surgical wounds but has no arterial class at all — arterial ulcers are
the hardest and clinically most important distinction in my original research question, and
your dataset is the most rigorously labeled source I have found for it. I would use it to run a
second, clearly-separated venous-vs-arterial experiment alongside my existing AZH-based model,
not merge it silently into a single dataset.

I would be very grateful if you could let me know:
1. Whether the dataset (or a de-identified subset) is available for use in an academic/research
   context outside your institution, and under what conditions (data use agreement, IRB
   documentation, etc.).
2. Any restrictions on class balance, image count, or resolution I should expect if access is
   granted.

I am happy to provide more detail on my project, my institution, or any documentation you would
need to evaluate this request.

Thank you for your time and for your paper's work on this problem.

Best regards,
[Your name]
[Your affiliation / institution]
[Your email — shubhi152006@gmail.com]

---

**Notes for you before sending:**
- Fill in your real name/affiliation — an anonymous or placeholder-looking request is less
  likely to get a reply from a clinical researcher.
- If you don't have an institutional affiliation, that's fine to say plainly ("independent
  researcher") — better than leaving it vague.
- No guarantee of a reply or of access; this project's report already treats "arterial class
  missing" as a stated limitation regardless of outcome, so this doesn't block anything else.
- If she does grant access, come back and I'll wire up a second venous-vs-arterial experiment
  using `src/data_prep.py` / `src/classical_baseline.py` on the new data, kept separate from the
  AZH 4-class model per the plan already in `paper/methodology/dataset_selection.md`.
