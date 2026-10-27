# Coding Standards

Read during review. The pre-commit hooks enforce formatting and lint; these are the judgement calls it cannot.

## Glossary terms

`GLOSSARY.md` terms, and its _Avoid_ lists, govern prose: docs, docstrings, comments, config keys, log and metadata keys. Code identifiers may name the **role** an element plays rather than the element: `receivers` (the transducers recording a shot), `src_idx` (the transmitting transducer's shot index), `u_recv`. Use the glossary term when the identifier means the element itself (`transducer_ij`, `n_transducers`).
