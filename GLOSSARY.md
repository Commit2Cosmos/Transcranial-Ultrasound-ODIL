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
One question about a single isolated parameter or feature, answered by one or more runs.
_Avoid_: run, when one execution is meant

**Experiment record**:
The short write-up of one experiment, kept in a bank for quick reference: its question, setup, results, the conclusions it does and does not support, and what has since invalidated it.
_Avoid_: report, results file

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
_Avoid_: sponge, absorbing boundary

**PDE term**:
The part of the objective that penalises the wave-equation residual over the interior and the PML, measured relative to the size of the wave term itself.

**Data term**:
The part of the objective that penalises the mismatch between the wavefield at the receivers and the observed data.

**Same physical problem**:
Two discretisations with the same domain, source shape and PML thickness, all measured in metres, differing only in grid spacing.

**Grid independence**:
The property that the balance between the objective's terms does not change when the same physical problem is discretised at a different resolution.
