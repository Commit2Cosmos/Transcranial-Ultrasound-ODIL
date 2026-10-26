# Transcranial Ultrasound ODIL

Frequency-domain full-waveform inversion of sound speed with a joint ODIL solve over wavefield and model.

## Language

### Configuration

**Default config**:
The single, complete set of experiment values that every run starts from.
_Avoid_: base config, default YAML, template config

**Experiment config**:
The changes one experiment makes to the default config, naming only the values that differ from it.
_Avoid_: user config, overlay, override file

**Override**:
A single value change given on the command line, applied on top of the experiment config.

**Resolved config**:
The fully specified config of one run: the default config with that run's changes applied, recorded with the run.
_Avoid_: effective config, final config

### Inversion

**Inverse crime**:
Generating the observed data with the same discrete operator and grid that the inversion uses, so that the true model fits the data exactly.
