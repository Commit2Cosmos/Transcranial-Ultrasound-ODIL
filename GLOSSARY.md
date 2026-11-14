# Transcranial Ultrasound ODIL

Frequency-domain full-waveform inversion of sound speed with a joint ODIL solve over wavefield and model.

## Language

### Configuration

**Default config**:
The single, complete set of experiment values that every run starts from.
_Avoid_: base config, default YAML, template config

**Run config**:
The changes one run makes to the default config, naming only the values that differ from it.
_Avoid_: experiment config, user config, overlay, override file

**Override**:
A single value change given on the command line, applied on top of the run config.

**Resolved config**:
The fully specified config of one run: the default config with that run's changes applied, recorded with the run.
_Avoid_: effective config, final config

### Experiments

**Run**:
One execution of the solver with one resolved config.

**Experiment**:
One question about one or more parameters or features, answered by one or more runs.
_Avoid_: run, when one execution is meant

**Protocol**:
The pre-registered plan of an experiment (its question, hypothesis and design), written before any run. Its design is made executable by the experiment spec.
_Avoid_: plan, proposal

**Experiment spec**:
The committed file that defines exactly the runs an experiment's protocol designs: its baseline, and the parameters or features it varies with the values each takes, every combination a run.
_Avoid_: experiment config, sweep file

**Experiment id**:
The number and short name reserved for an experiment when it is launched, shared by its spec, its runs' directory and its record.

**Scratch run**:
A run made from code with uncommitted changes, for trying things out. No experiment record may cite it.

**Probe run**:
A few iterations of an experiment's baseline, run only to estimate how long the experiment will take.
_Avoid_: test run, dry run

**Baseline**:
The run config an experiment varies its parameters or features from. A follow-up experiment names its own baseline, which may come from an earlier experiment's findings.
_Avoid_: base config, reference config; baseline for the starting model of a run

**Experiment chain**:
A sequence of experiments in which each one's findings set the next one's baseline.
_Avoid_: series, campaign

**Start row**:
The metrics recorded at the start of each frequency band, after the band's warm start and before its first update.
_Avoid_: baseline row

**Experiment record**:
The short write-up of one experiment, kept in a bank for quick reference: its question, setup, results, the conclusions it does and does not support, and whether it has since been invalidated.
_Avoid_: report, results file

**Experiment bank**:
All experiment records together, with an index that an agent reads first.
_Avoid_: results archive, log

**Invalidated**:
Said of an experiment record when a later code change means one of its supported conclusions no longer holds.

**Voided**:
Said of a launch whose runs no longer test its protocol, so it is cancelled before any record of it is banked and its id released for reuse.
_Avoid_: aborted (an aborted experiment ran its protocol and is recorded)

### Inversion

**Inverse crime**:
Generating the observed data with the same discrete operator and grid that the inversion uses, so that the true model fits the data exactly.

**Transducer**:
An element on the ring that both transmits and records. Every transducer transmits exactly once per run and records every shot except its own.
_Avoid_: source or receiver alone, when the element is meant

**Shot**:
One transmission by one transducer, recorded by all the other transducers.

**Offset**:
The separation along the ring between the transmitting transducer and a recording transducer.

**Trace**:
What one transducer records of one shot at one frequency.

**Interior**:
The physical region being imaged; everything on the grid except the PML.
_Avoid_: active region, domain

**Head region**:
The skull and everything it encloses.

**Skull region**:
The skull alone: the bony shell of the head region.

**Brain region**:
Everything inside the skull, excluding the skull itself.
_Avoid_: interior, intracranial ROI, head ROI

**PML**:
The absorbing layer around the interior that suppresses reflections from the grid boundary. Its size that matters physically is its thickness in metres.
_Avoid_: sponge, absorbing boundary, PML ring (ring is the transducer ring)

**PDE term**:
The part of the objective that penalises the wave-equation residual over the interior and the PML, measured relative to the size of the wave term itself.

**Data term**:
The part of the objective that penalises the mismatch between the wavefield at the receivers and the observed data.

**Trace weighting**:
How much the data term counts each trace's misfit relative to the other traces. Separate from the data weight, which balances the data term against the other terms.
_Avoid_: data normalisation

**Regularisation term**:
The part of the objective that penalises roughness of the velocity model, averaged over the interior in dimensionless units so that its size does not depend on the grid resolution.
_Avoid_: penalty, smoothing term

**Wavefield block**:
The wavefield unknowns of the joint objective: one complex field per shot and frequency.
_Also_: u-block

**Model block**:
The sound-speed unknowns of the joint objective.
_Also_: c-block
_Avoid_: z block

**Wavefield preconditioner**:
A change of variables that the joint solve optimises in place of the wavefield block, to make it converge in fewer iterations.
_Avoid_: wavefield scaling

**Model preconditioner**:
A change of variables that the joint solve optimises in place of the model block, to balance its curvature against the wavefield block's.
_Avoid_: model scaling

**Optimised variable**:
The variable the joint solve updates in place of a block's own unknowns, with or without a preconditioner. The model block's latent and the wavefield block's scaling are not preconditioners: they are always on, and they keep the velocity in its bounds and set the block's units.

**True model**:
The velocity model the observed data is generated from.

**True wavefield**:
The wavefield the true model produces over the whole grid. Under the inverse crime it matches the observed data exactly at the transducers.

**Initial model**:
The velocity model a band starts from: the run's configured initial model for the first band, the previous band's final model after that.
_Avoid_: starting model, baseline

**Initial wavefield**:
The values the wavefield block holds before a band's first update.
_Avoid_: starting wavefield

**Warm start**:
Seeding the initial wavefield by solving the wave equation on the band's initial model. Every band currently begins with one.

**Same physical problem**:
Two discretisations with the same domain, source shape and PML thickness, all measured in metres, differing only in grid spacing.

**Grid independence**:
The property that the balance between the objective's terms does not change when the same physical problem is discretised at a different resolution, or when only the PML thickness changes.
